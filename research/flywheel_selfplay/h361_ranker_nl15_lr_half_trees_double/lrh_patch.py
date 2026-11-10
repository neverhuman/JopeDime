"""NL15-preserving shrinkage: halve ranker LR and double all iteration limits."""
import ast,copy,hashlib,inspect,textwrap
import leaves_patch
LR_SCALE=.5;ITER_SCALE=2
def wrap(value,scale):return ast.BinOp(left=value,op=ast.Mult(),right=ast.Constant(scale))
def unwrap(value,scale):
 assert isinstance(value,ast.BinOp) and isinstance(value.op,ast.Mult) and ast.literal_eval(value.right)==scale
 return value.left
def transform(source,role):
 if role=='constructor':_,baseline=leaves_patch.transform(source)
 else:baseline=ast.parse(textwrap.dedent(source))
 tree=copy.deepcopy(baseline)
 if role=='constructor':
  call=leaves_patch.calls(tree)[0];kw={x.arg:x for x in call.keywords}
  assert isinstance(kw['learning_rate'].value,ast.Call) and isinstance(kw['learning_rate'].value.func,ast.Name) and kw['learning_rate'].value.func.id=='float'
  kw['learning_rate'].value=wrap(kw['learning_rate'].value,LR_SCALE);kw['n_estimators'].value=wrap(kw['n_estimators'].value,ITER_SCALE)
 else:
  count=0
  for call in [n for n in ast.walk(tree) if isinstance(n,ast.Call)]:
   match=(role=='veto_selection' and isinstance(call.func,ast.Attribute) and call.func.attr=='predict') or (role=='member_lock' and isinstance(call.func,ast.Attribute) and call.func.attr=='verify_reload')
   if match:
    for kw in call.keywords:
     if kw.arg=='num_iteration' and not (isinstance(kw.value,ast.Constant) and kw.value.value is None):kw.value=wrap(kw.value,ITER_SCALE);count+=1
  assert count==2,(role,count)
  if role=='member_lock':
   changed=0
   for node in [n for n in ast.walk(tree) if isinstance(n,ast.Dict)]:
    for i,key in enumerate(node.keys):
     if isinstance(key,ast.Constant) and key.value in ['selected_lr','selected_n_estimators']:
      node.values[i]=wrap(node.values[i],LR_SCALE if key.value=='selected_lr' else ITER_SCALE);changed+=1
   assert changed==2
 ast.fix_missing_locations(tree);assert ast.dump(baseline,include_attributes=False)==ast.dump(strip(tree,role),include_attributes=False)
 return baseline,tree
def strip(tree,role):
 tree=copy.deepcopy(tree)
 if role=='constructor':
  kw={x.arg:x for x in leaves_patch.calls(tree)[0].keywords};kw['learning_rate'].value=unwrap(kw['learning_rate'].value,LR_SCALE);kw['n_estimators'].value=unwrap(kw['n_estimators'].value,ITER_SCALE)
 else:
  for call in [n for n in ast.walk(tree) if isinstance(n,ast.Call)]:
   match=(role=='veto_selection' and isinstance(call.func,ast.Attribute) and call.func.attr=='predict') or (role=='member_lock' and isinstance(call.func,ast.Attribute) and call.func.attr=='verify_reload')
   if match:
    for kw in call.keywords:
     if kw.arg=='num_iteration' and not (isinstance(kw.value,ast.Constant) and kw.value.value is None):kw.value=unwrap(kw.value,ITER_SCALE)
  if role=='member_lock':
   for node in [n for n in ast.walk(tree) if isinstance(n,ast.Dict)]:
    for i,key in enumerate(node.keys):
     if isinstance(key,ast.Constant) and key.value in ['selected_lr','selected_n_estimators']:node.values[i]=unwrap(node.values[i],LR_SCALE if key.value=='selected_lr' else ITER_SCALE)
 return tree
def patched_function(fun,role):
 source=inspect.getsource(fun);old,new=transform(source,role);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,role=role,H352_baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),inverse_H352_exact=True)
def install(G):
 records=[]
 for mod,name,role in [('topk','fit_ranker','constructor'),('bagged','fit_ranker_n','constructor'),('veto','run_cell_veto','veto_selection'),('r302','fit_and_freeze_member','member_lock')]:
  fun,r=patched_function(getattr(G[mod],name),role);setattr(G[mod],name,fun);records.append(r)
 return dict(status='PASS',functions=records,num_leaves=15,max_depth=3,learning_rate_scale=LR_SCALE,ranker_iterations_scale=ITER_SCALE,seed_grid_labels_unchanged=True,veto_unchanged=True)
