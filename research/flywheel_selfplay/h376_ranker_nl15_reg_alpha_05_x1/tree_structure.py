"""Read actual per-tree edge depth/leaf count from existing native model text; zero fits."""
import re
from collections import Counter

def parse_trees(text):
 pieces=re.split(r'^Tree=(\d+)\n',text,flags=re.M);assert len(pieces)>1 and len(pieces)%2==1
 result=[]
 for i in range(1,len(pieces),2):
  assert int(pieces[i])==len(result);body=pieces[i+1]
  def field(name):
   match=re.search(r'^'+name+r'=(.*)$',body,re.M);assert match is not None,name
   return match.group(1).strip()
  leaves=int(field('num_leaves'));left=list(map(int,field('left_child').split()));right=list(map(int,field('right_child').split()))
  assert leaves>=1 and len(left)==len(right)==leaves-1
  if leaves==1:result.append([0,1]);continue
  seen_nodes=set();seen_leaves=set()
  def visit(node,depth):
   if node<0:
    leaf=-node-1;assert 0<=leaf<leaves and leaf not in seen_leaves;seen_leaves.add(leaf);return depth
   assert 0<=node<len(left) and node not in seen_nodes;seen_nodes.add(node)
   return max(visit(left[node],depth+1),visit(right[node],depth+1))
  depth=visit(0,0);assert len(seen_nodes)==leaves-1 and len(seen_leaves)==leaves
  result.append([depth,leaves])
 return result

def validate(trees,max_depth,num_leaves):
 assert trees
 for depth,leaves in trees:
  assert isinstance(depth,int) and isinstance(leaves,int) and depth>=0 and 1<=leaves<=num_leaves and (max_depth<=0 or depth<=max_depth)
 return trees

def summarize(trees):
 assert trees
 depths=Counter(d for d,l in trees);leaves=Counter(l for d,l in trees)
 return dict(trees=len(trees),max_actual_depth=max(depths),max_actual_leaves=max(leaves),depth_histogram={str(k):v for k,v in sorted(depths.items())},leaf_histogram={str(k):v for k,v in sorted(leaves.items())})
