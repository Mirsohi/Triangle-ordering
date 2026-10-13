#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
// Exact monotone insertion search. Only u changes score, so only its incident
// edge orientations change. Recompute all work terms incident to affected
// vertices before accepting. Scores always define a valid total order.
extern "C" int64_t dove_refine(int64_t n,int64_t m,const int32_t* edges,
 double* score,uint64_t seed,double seconds,int64_t max_proposals,int64_t* info) {
 using clock=std::chrono::steady_clock;
 std::vector<std::vector<int64_t>> adj(n);
 std::vector<int64_t> d(n,0),change(n,0),seen(m,0);
 auto before=[&](int64_t u,int64_t v,double su,double sv){return su<sv||(su==sv&&u<v);};
 for(int64_t e=0;e<m;++e){int64_t u=edges[2*e],v=edges[2*e+1];
  adj[u].push_back(e);adj[v].push_back(e);++d[before(u,v,score[u],score[v])?u:v];}
 int64_t work=0;
 for(int64_t e=0;e<m;++e)work+=std::min(d[edges[2*e]],d[edges[2*e+1]]);
 std::mt19937_64 rng(seed);int64_t accepted=0,proposals=0;
 auto start=clock::now();
 for(;proposals<max_proposals;++proposals){
  if(std::chrono::duration<double>(clock::now()-start).count()>=seconds)break;
  int64_t e=rng()%m,u=edges[2*e+(rng()%2)],v=edges[2*e]+edges[2*e+1]-u;
  double news=std::nextafter(score[v],rng()%2?INFINITY:-INFINITY);
  if(!std::isfinite(news)||news==score[u])continue;
  std::vector<int64_t> affected{u},terms;
  for(auto f:adj[u]){int64_t w=edges[2*f]+edges[2*f+1]-u;
   bool old=before(u,w,score[u],score[w]),now=before(u,w,news,score[w]);
   if(old!=now){change[u]+=now?1:-1;change[w]+=now?-1:1;affected.push_back(w);}}
  for(auto x:affected)for(auto f:adj[x])if(seen[f]!=proposals+1){seen[f]=proposals+1;terms.push_back(f);}
  int64_t delta=0;
  for(auto f:terms){int64_t a=edges[2*f],b=edges[2*f+1];
   delta+=std::min(d[a]+change[a],d[b]+change[b])-std::min(d[a],d[b]);}
  if(delta<0){score[u]=news;work+=delta;++accepted;for(auto x:affected)d[x]+=change[x];}
  for(auto x:affected)change[x]=0;
 }
 info[0]=proposals;info[1]=accepted;return work;
}
// Adjacent swaps: after 10 sweeps, count every remaining improving adjacent
// pair. A zero count is a certificate for this neighborhood only.
extern "C" int64_t dove_adjacent(int64_t n,int64_t m,const int32_t* edges,
 int64_t* order,int64_t rounds,int64_t* info){
 std::vector<int64_t> rank(n),d(n,0);std::vector<std::vector<int64_t>> adj(n);
 for(int64_t i=0;i<n;++i)rank[order[i]]=i;
 for(int64_t e=0;e<m;++e){auto u=edges[2*e],v=edges[2*e+1];
  adj[u].push_back(v);adj[v].push_back(u);++d[rank[u]<rank[v]?u:v];}
 auto delta=[&](int64_t u,int64_t v){
  bool edge=false;for(auto w:adj[u])if(w==v){edge=true;break;}
  if(!edge)return int64_t(0);
  int64_t a=d[u],b=d[v],diff=std::min(a-1,b+1)-std::min(a,b);
  for(auto x:adj[u])if(x!=v&&d[x]>=a)--diff;
  for(auto x:adj[v])if(x!=u&&d[x]>=b+1)++diff;
  return diff;};
 int64_t accepted=0;
 for(int64_t r=0;r<rounds;++r){int64_t count=0;
  for(int parity=0;parity<2;++parity)for(int64_t i=parity;i+1<n;i+=2){
   auto u=order[i],v=order[i+1];if(delta(u,v)<0){--d[u];++d[v];std::swap(order[i],order[i+1]);++count;}}
  accepted+=count;if(!count)break;
 }
 int64_t work=0,improving=0;
 for(int64_t e=0;e<m;++e)work+=std::min(d[edges[2*e]],d[edges[2*e+1]]);
 for(int64_t i=0;i+1<n;++i)improving+=delta(order[i],order[i+1])<0;
 info[0]=accepted;info[1]=improving;return work;
}
