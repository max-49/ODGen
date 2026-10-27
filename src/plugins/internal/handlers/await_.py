import os
from src.plugins.handler import Handler
from .functions import call_function
from src.core.utils import NodeHandleResult, ExtraInfo
from src.core.logger import loggers
from src.plugins.internal.utils import get_df_callback, to_obj_nodes

# --- CSEC 759 A3 (Massimo Marino): synchronous await resolution --------------
# The esprima front end emits AST_YIELD for `await`, `yield` and spread (`...x`);
# only the 'JS_AWAIT_EXPRESSION' flag identifies a real await. The baseline
# handler resolves every operand through the modeled Promise.prototype.then,
# whose callback runs from the microtask queue after the entry function has
# finished, so the await expression itself evaluates to no objects (A2 finding).
#
# ODGEN_AWAIT_MODE selects the behaviour (used for the A3 ablation):
#   full            : B1 + B2 (default)
#   settled_only    : B1 only   (B2 disabled)
#   nonpromise_only : B2 only   (B1 disabled)
#   off             : original artifact behaviour for every operand
# B1: operand is a promise that already has `fulfilled_with`  -> its value now
# B2: operand is not a promise (await v == await Promise.resolve(v)) -> v itself
# B3: anything else (pending promise, or a disabled branch)    -> original path
AWAIT_MODE = os.environ.get('ODGEN_AWAIT_MODE', 'full')
_VALID_MODES = ('full', 'settled_only', 'nonpromise_only', 'off')
if AWAIT_MODE not in _VALID_MODES:
    raise ValueError('ODGEN_AWAIT_MODE must be one of %s, got %r'
                     % (_VALID_MODES, AWAIT_MODE))
_MAX_FLATTEN_DEPTH = 8  # promise resolved with a promise: follow at most 8 levels
_PROMISE_ATTRS = ('is_promise', 'executors', 'fulfilled_with', 'rejected_with')


def is_modeled_promise(G, obj):
    """True if obj was created by ODGen's Promise model.
    new Promise(...) / Promise.resolve(...) link __proto__ -> G.promise_prototype
    and set 'executors'. Promise.prototype.then/catch return a bare object with
    no __proto__ and no attributes until their microtask runs; v1 missed these
    and treated them as plain values (A3 decision D-A3-2, case N2s), so v2 marks
    them with 'is_promise' in modeled_js_builtins.promise_p_then/catch."""
    attrs = G.get_node_attr(obj)
    if any(a in attrs for a in _PROMISE_ATTRS):
        return True
    return G.promise_prototype in G.get_prop_obj_nodes(obj, '__proto__')


def resolve_settled(G, promise, ast_node, depth=0):
    """Objects a settled promise was fulfilled with, flattening nested
    promises (JS adopts the state of a promise resolved with a promise).
    Returns None if the chain reaches a promise that is still pending."""
    value = G.get_node_attr(promise).get('fulfilled_with')
    if value is None:
        return None
    result = []
    for obj in to_obj_nodes(G, value, ast_node):
        if depth < _MAX_FLATTEN_DEPTH and is_modeled_promise(G, obj):
            inner = resolve_settled(G, obj, ast_node, depth + 1)
            if inner is None:
                return None
            result.extend(inner)
        else:
            result.append(obj)
    return result


class HandleAwait(Handler):
    """
    the handler to handle await(AST_YIELD)
    """
    def __init__(self, G, node_id, extra=None):
        self.G = G
        self.node_id = node_id
        self.extra = extra

    def process(self):
        from src.plugins.manager_instance import internal_manager
        promises = internal_manager.dispatch_node(self.G.get_ordered_ast_child_nodes(self.node_id)[0])
        flag = self.G.get_node_attr(self.node_id).get('flags:string[]')
        if AWAIT_MODE == 'off' or flag != 'JS_AWAIT_EXPRESSION':
            # spread / yield, or the switch is off: original artifact code path
            return self._then_path(promises, promises.obj_nodes)

        use_b1 = AWAIT_MODE in ('full', 'settled_only')
        use_b2 = AWAIT_MODE in ('full', 'nonpromise_only')
        resolved = []   # objects the await expression evaluates to now
        pending = []    # operands left to the original then-callback path
        used = []       # objects the await reads (for def-use data flow)
        # to_obj_nodes also turns literal operands (await 'x') into objects
        for obj in to_obj_nodes(self.G, promises, self.node_id):
            used.append(obj)
            if is_modeled_promise(self.G, obj):
                value = resolve_settled(self.G, obj, self.node_id) if use_b1 else None
                if value is not None:
                    branch = 'B1_settled'
                    resolved.extend(value)
                    used.extend(value)
                else:
                    branch = 'B3_pending' if use_b1 else 'B3_b1_disabled'
                    pending.append(obj)
            elif use_b2:
                branch = 'B2_nonpromise'
                resolved.append(obj)
            else:
                branch = 'B3_b2_disabled'
                pending.append(obj)
            loggers.main_logger.info('AWAIT-SYNC mode={} ast={} line={} obj={} branch={}'.format(
                AWAIT_MODE, self.node_id,
                self.G.get_node_attr(self.node_id).get('lineno:int'), obj, branch))

        if pending:
            result = self._then_path(promises, pending)
            resolved.extend(result.obj_nodes)
        return NodeHandleResult(ast_node=self.node_id,
                                obj_nodes=list(dict.fromkeys(resolved)),
                                used_objs=list(dict.fromkeys(used)),
                                callback=get_df_callback(self.G))

    def _then_path(self, promises, operands):
        """Original artifact behaviour (unchanged logic, moved into a method):
        register a Python onFulfilled callback via the modeled
        Promise.prototype.then for each operand and return whatever it has
        collected so far (usually nothing; it runs from the microtask queue)."""
        returned_objs = set()
        # prepare a callback (onFulfilled) function
        def python_callback(
                G, caller_ast, extra, _, value=NodeHandleResult(), *args):
            nonlocal returned_objs
            returned_objs.update(to_obj_nodes(G, value, self.node_id))
            return NodeHandleResult()
        cb = self.G.add_blank_func_with_og_nodes(
            'pythonCallback', python_func=python_callback)
        # call promise.then
        for promise in operands:
            then = self.G.get_prop_obj_nodes(
                        self.G.promise_prototype, 'then', self.extra.branches)
            result, _ = call_function(self.G, then,
                    args=[NodeHandleResult(obj_nodes=[cb])],
                    this=NodeHandleResult(obj_nodes=[promise]),
                    extra=self.extra, caller_ast=self.node_id)
        # add control flows from the callback function's EXIT node to
        # the current statement
        cb_ast = self.G.get_obj_def_ast_node(cb)
        exit_node = self.G.get_successors(cb_ast, edge_type='EXIT')[0]
        self.G.add_edge_if_not_exist(exit_node,
            self.G.find_nearest_upper_CPG_node(self.node_id), {"type:TYPE": "FLOWS_TO"})
        return NodeHandleResult(ast_node=self.node_id,
                                obj_nodes=list(returned_objs),
                                used_objs=promises.obj_nodes,
                                callback=get_df_callback(self.G))
