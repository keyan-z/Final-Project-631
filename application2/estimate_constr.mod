# This file estimates parameters using a constrained approach.

# Parameters
param N;		  		#no. of firms
param beta;    			#discount factor
param euler_const;		
param NS;				#no. of states for market size
param NM;				#no. of markets
param NT;				#no. of periods

# Sets
set I :=1..N;			#index set of firms
set M :=1..NM;			#index set of markets
set T :=1..NT;			#index set of periods
set A :=0..1;			#index set of players' action space
set S	 :=1..NS;			#index set of market sizes
set Aset := 0..2**N - 1;	#Action space of firms: 2^N profiles
set Apow {k in Aset} := {i in I: (k div 2**(i-1)) mod 2 = 1};
set X:= {S,Aset};		#State space

# Data
param ObsA {I,M,T};		#decision of firm i in market m at t
param ObsXS {M,T};		#market size of market m at time t
param NObsX_a1 {I,X};
param NObsX_a0 {I,X};

#infer from data: convert t-1 action profile into Aset index
param ObsXA {m in M,t in T} := if t>1 then sum {i in I} ((2**(i-1))*ObsA[i,m,t-1]) else 0;

#infer from data: set of (markets m, period t) in each state
set MktX {(xs,xa) in X} := {m in M, t in T: ObsXS[m,t]=xs and ObsXA[m,t]=xa};

param S_trans {snext in S,s in S};

# Structural Parameters
var theta_FC_hat {i in I} >= 0;	#firm-specific fixed cost
var theta_RS_hat >= 0;		#present state-dependent utility
var theta_EC_hat >= 0;		#entry cost based on t-1 state
var theta_RN_hat >= 0;		#competitor-dependent utility

# Starting Values
param theta_FC {i in I};
param theta_RS;
param theta_EC;
param theta_RN;

param P_start {I,X}; # probability of a=1 
param V_start {I,X};

# Variables
var V {I,X} ;     #value functions
var P {I,X,A} >=0, <=1;
var fP {(xsnext,xanext) in X, (xs,xa) in X} = S_trans[xsnext,xs]*(prod {i in I} P[i,xs,xa,if i in Apow[xanext] then 1 else 0]);

var fPchoice {i in I, (xsnext,xanext) in X, (xs,xa) in X, a in A} = if (if i in Apow[xanext] then 1 else 0)==a then fP[xsnext,xanext,xs,xa]/P[i,xs,xa,if i in Apow[xanext] then 1 else 0];

var e {i in I,(xs,xa) in X,a in A} = euler_const - log(P[i,xs,xa,a]); #assuming sigma=1

# Per-period payoff: Conditional on present/past actions, state
var payoffpp {i in I, xs in S, aa in Aset, aprev in A} = 
if i in Apow[aa] then theta_RS_hat*xs - theta_RN_hat*log(1+card(Apow[aa])-1) - theta_FC_hat[i] - theta_EC_hat*(1-aprev) else 0;

#Expected payoff: Condition on firm i's present action and state
var epayoff {i in I, (xs,xa) in X, a in A} = sum {aanext in Aset:(if i in Apow[aanext] then 1 else 0)==a} ((prod {j in I:i<>j} P[j,xs,xa,if j in Apow[aanext] then 1 else 0])*payoffpp[i,xs,aanext,if i in Apow[xa] then 1 else 0]);

#Choice-specific value functions: Condition on action and state
var Vchoice {i in I, (xs,xa) in X, a in A} = epayoff[i,xs,xa,a]+beta*sum { (xsnext,xanext) in X} V[i,xsnext,xanext]*fPchoice[i,xsnext,xanext,xs,xa,a];


#Optimization Setup

maximize LogLikelihood: 

	sum {i in I, (xs,xa) in X}  (NObsX_a1[i,xs,xa]*log(P[i,xs,xa,1])+NObsX_a0[i,xs,xa]*log(P[i,xs,xa,0]));

subject to

Bellman {i in I, (xs,xa) in X}:
V[i,xs,xa]= sum {a in A} P[i,xs,xa,a]*(epayoff[i,xs,xa,a] + e[i,xs,xa,a]) + beta*(sum {(xsnext,xanext) in X} V[i,xsnext,xanext]*fP[xsnext,xanext,xs,xa]);

CondChoiceProb {i in I, (xs,xa) in X}: 
P[i,xs,xa,1]=exp(Vchoice[i,xs,xa,1])/(sum {a in A} exp(Vchoice[i,xs,xa,a]));

ProbSum {i in I, (xs,xa) in X}: sum {a in A} P[i,xs,xa,a]=1;


#Optimization

problem AM07:

#Objective function
LogLikelihood,

#Variables
V,P,fP,fPchoice,e,Vchoice,payoffpp,epayoff,
theta_FC_hat, theta_RS_hat, theta_EC_hat, theta_RN_hat,

#Constraints
Bellman,CondChoiceProb, ProbSum;