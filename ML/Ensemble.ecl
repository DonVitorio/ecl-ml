﻿IMPORT * FROM ML;
IMPORT * FROM ML.Types;

EXPORT Ensemble := MODULE
  EXPORT modelD_Map :=	DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},{'value','4'},{'new_node_id','5'},{'support','6'},{'group_id','7'}], {STRING orig_name; STRING assigned_name;});
  EXPORT STRING modelD_fields := 'node_id,level,number,value,new_node_id,support,group_id';	// need to use field map to call FromField later
  EXPORT modelC_Map :=	DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},{'value','4'},{'high_fork','5'},{'new_node_id','6'},{'group_id','7'}], {STRING orig_name; STRING assigned_name;});
  EXPORT STRING modelC_fields := 'node_id,level,number,value,high_fork,new_node_id,group_id';	// need to use field map to call FromField later

  // Splitting Results Data Structures
  EXPORT gSplitD := RECORD
    Trees.SplitD;
    t_Count       group_id;     // Tree Number Identifier
  END;
  EXPORT gSplitC := RECORD
    Trees.SplitC;
    t_Count       group_id;     // Tree Number Identifier
  END;
  
  // Learning Data Structures - Internal
  SHARED gNodeInstDisc := RECORD
    NodeID;
    DiscreteField;              // Instance Independent Data - one discrete attribute
    t_Discrete    depend;       // Instance Dependant value
    t_Count       support:=0;   // Support during learning
    t_Count       group_id;     // Tree Number Identifier
  END;
  SHARED gNodeInstCont := RECORD
    NodeID;
    NumericField;               // Instance Independent Data - one numeric attribute
    t_Discrete    depend;       // Instance Dependant value
    BOOLEAN       high_fork:= FALSE; // Fork Flag - Low: lower or equal than value, High: greater than value
    t_Count       group_id;
  END;
  SHARED DepGroupedRec := RECORD(DiscreteField)
    UNSIGNED      group_id := 0;
    t_RecordID    new_id := 0;
  END;

  // Learning TRANSFORMs and FUNCTIONs - Internal
  SHARED DepGroupedRec GroupDepRecords (DiscreteField l, Sampling.idListGroupRec r) := TRANSFORM
    SELF.group_id := r.gNum;
    SELF.new_id   := r.id;
    SELF          := l;
  END;
  SHARED NxKoutofM(t_Count N, t_FieldNumber K, t_FieldNumber M) := FUNCTION
    rndFeatRec:= RECORD
      t_Count	      gNum   :=0;
      t_FieldNumber number :=0;
      t_FieldReal   rnd    :=0;
    END;
    seed:= DATASET([{0,0,0}], rndFeatRec);
    group_seed := DISTRIBUTE(NORMALIZE(seed, N,TRANSFORM(rndFeatRec, SELF.gNum:= COUNTER)), gNum);
    allFields  := NORMALIZE(group_seed, M, TRANSFORM(rndFeatRec, SELF.number:= (COUNTER % M) +1, SELF.rnd:=RANDOM(), SELF:=LEFT),LOCAL);
    allSorted  := SORT(allFields, gNum, rnd, LOCAL);
    raw_set    := ENTH(allSorted, K, M, 1);
    RETURN TABLE(raw_set, {gNum, number});
  END;
  SHARED DiscreteField GetDRecords(DiscreteField l, Sampling.idListGroupRec r) := TRANSFORM
    SELF.id := r.id;
    SELF.number := l.number;
    SELF.value := l.value;
  END;
  SHARED NumericField GetCRecords(NumericField l, Sampling.idListGroupRec r) := TRANSFORM
    SELF.id := r.id;
    SELF.number := l.number;
    SELF.value := l.value;
  END;
  SHARED gNodeInstDisc GenerateDiscRoots(DiscreteField dep, Sampling.idListGroupRec depSample) := TRANSFORM
    SELF.group_id := depSample.gNum;
    SELF.node_id  := depSample.gNum;
    SELF.level    := 1;
    SELF.id       := depSample.id;
    SELF.number   := 1;
    SELF.value    := dep.value;
    SELF.depend   := dep.value;
  END;
  SHARED gNodeInstCont GenerateContRoots(DiscreteField dep, Sampling.idListGroupRec depSample) := TRANSFORM
    SELF.group_id := depSample.gNum;
    SELF.node_id  := depSample.gNum;
    SELF.level    := 1;
    SELF.id       := depSample.id;
    SELF.number   := 1;
    SELF.value    := dep.value;
    SELF.depend   := dep.value;
  END;
/* Discrete implementation*/
// Function to split a set of nodes based on Feature Selection and Gini Impurity, used in Random Forest Classifier Discrete Learning,
// the nodes received were generated sampling with replacement nTrees times.
// Note: returns treeNum Decision Trees, split based on Gini Impurity
//       selects fsNum out of total number of features, they must start at 1 and cannot exist a gap in the numeration.
//       Gini Impurity's default parameters: Purity = 1.0 and maxLevel (Depth) = 32 (up to 126 max iterations)
// more info http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#overview
  EXPORT SplitFeatureSampleGI(DATASET(DiscreteField) Indep, DATASET(DiscreteField) Dep, t_Count treeNum, t_Count fsNum, REAL Purity=1.0, t_level maxLevel=32) := FUNCTION
    N       := MAX(Dep, id);       // Number of Instances
    totFeat := COUNT(Indep(id=N)); // Number of Features
    depth   := MIN(255, maxLevel); // Max number of iterations when building trees (max 256 levels)
    dDep    := DISTRIBUTE(Dep   , HASH32(Id));
    dIndep  := DISTRIBUTE(Indep , HASH32(id));
    // sampling with replacement the original dataset to generate treeNum Datasets
    grList0   := Sampling.GenerateNSampleList(treeNum, N);  // mapping list between new group-instances and original instances
    dgrLstOld := DISTRIBUTE(grList0  , HASH32(oldId));      // oldId holds the correspondent original instance id
    // for each tree assign all their sampled instances at root node
    roots     := JOIN(dDep  , dgrLstOld, LEFT.id = RIGHT.oldId, GenerateDiscRoots(LEFT, RIGHT), LOCAL);
    // Calculated only once, used at each iteration inside loopbody
    all_Data0 := JOIN(dIndep, dgrLstOld, LEFT.id = RIGHT.oldId, GetDRecords(LEFT, RIGHT), LOCAL);
    all_Data  := DISTRIBUTE(all_Data0, HASH32(id));         // distributed by sampled id
    // loopbody function
    gNodeInstDisc RndFeatSelPartitionGRBased(DATASET(gNodeInstDisc) nodes, t_Count p_level):= FUNCTION
      dNodes := DISTRIBUTE(nodes, HASH32(group_id, node_id));
      // Calculating Gini Impurity for each node
      aggNode  := TABLE(dNodes,  {group_id, node_id, depend, cnt := COUNT(GROUP)}, group_id, node_id, depend, LOCAL);
      aggNodec := TABLE(aggNode, {group_id, node_id,     tcnt := SUM(GROUP, cnt)}, group_id, node_id,         LOCAL);
      recNode  := RECORD
        aggNode;
        REAL4 prop; // Proportion pertaining to its dependant value
      END;
      propNode   := JOIN(aggNode, aggNodec, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id, TRANSFORM(recNode, SELF.prop := LEFT.cnt/RIGHT.tcnt, SELF := LEFT), LOCAL);
      Purities   := TABLE(propNode, {group_id, node_id, totalCnt := SUM(GROUP, cnt), gini := 1-SUM(GROUP, prop*prop)}, group_id, node_id, LOCAL); // Compute gini purity for each node   
      // Filtering pure and non-pure nodes
      PureEnough := Purities(1-Purity >= gini); // Filtering pure nodes
      NotPure    := Purities(1-Purity <  gini); // Filtering pure nodes
      isPureNode := JOIN(dNodes, PureEnough, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id, 
                            TRANSFORM({gNodeInstDisc, BOOLEAN pure:= FALSE}, SELF.pure:=RIGHT.node_id>0, SELF.support:= RIGHT.totalCnt, SELF:=LEFT), LEFT OUTER, LOCAL);
      pass_thru   := PROJECT(isPureNode(pure = TRUE), gNodeInstDisc, LOCAL);
      ImpureNodes := PROJECT(isPureNode(pure = FALSE), gNodeInstDisc, LOCAL); // still distributted by group_id, node_id
      // Gather only the data needed for each LOOP iteration,
      NodexKoutofM(DATASET(RECORDOF(Purities)) topNodes, t_FieldNumber K, t_FieldNumber M) := FUNCTION  
        rndFeatRec:= RECORD
          t_Count       group_id  := 0;
          t_node        node_id   := 0;
          t_FieldNumber number    := 0;
          t_FieldReal   rnd       := 0;
        END;
        seed:= DATASET([{0,0,0,0}], rndFeatRec);
        node_seed := JOIN(topNodes, seed, TRUE, TRANSFORM(rndFeatRec, SELF:= LEFT), ALL); // one seed for each Node
        // generating K features per node map and sampling only K out of M per node
        allFields := NORMALIZE(node_seed, M, TRANSFORM(rndFeatRec, SELF.number:= (COUNTER % M) +1, SELF.rnd:=RANDOM(), SELF:=LEFT),LOCAL);
        allSorted := SORT(allFields, group_id, node_id, rnd, LOCAL);
        raw_set   := ENTH(allSorted, K, M, 1); // LOCAL doesn't give exact results
        RETURN TABLE(raw_set, {group_id, node_id, number});
      END;
      nodesFeatSet  := DISTRIBUTE(NodexKoutofM(NotPure, fsNum, totFeat), HASH32(group_id, node_id));  // generating list of features selected for each node and distributing to match NonPureNodes
      // Populating nodes' attributes to split
      ftSetInst := JOIN(ImpureNodes, nodesFeatSet, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id, TRANSFORM(gNodeInstDisc, SELF.number:= RIGHT.number, SELF:= LEFT), LOCAL);
      toSplit   := JOIN(all_Data, DISTRIBUTE(ftSetInst, HASH32(id)), LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(gNodeInstDisc, SELF.value:= LEFT.value; SELF:= RIGHT;), LOCAL);
      // new set of nodes-instance-att_value to find best split per node based on gini impurity
      this_set  := DISTRIBUTE(toSplit, HASH32(group_id, node_id));
      aggBranch := TABLE(this_set , {group_id, node_id, number, value, depend, Cnt := COUNT(GROUP)}, group_id, node_id, number, value, depend, LOCAL);
      aggBrCum  := TABLE(aggBranch, {group_id, node_id, number, value,     TCnt := SUM(GROUP, Cnt)}, group_id, node_id, number, value,         LOCAL);
      r := RECORD
        aggBranch;
        REAL4 Prop; // Proportion pertaining to its dependent value
      END;
      // Calculating Gini Impurity after every split
      prop      := JOIN(aggBranch, aggBrCum, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value,
                      TRANSFORM(r, SELF.Prop := LEFT.Cnt/RIGHT.Tcnt, SELF := LEFT), LOCAL);
      gini_per  := TABLE(prop, {group_id, node_id, number, value, tcnt := SUM(GROUP,Cnt),val := SUM(GROUP,Prop*Prop)}, group_id, node_id, number, value, LOCAL);
      gini      := TABLE(gini_per, {group_id, node_id, number, gini_t := SUM(GROUP,tcnt*val)/SUM(GROUP,tcnt)}, group_id, node_id, number, LOCAL);
      // Selecting the split with max Info Gain Ratio per node
      splits    := DEDUP(SORT(gini, group_id, node_id, -gini_t, LOCAL), group_id, node_id, LOCAL);
      // new split nodes found
      new_spl0  := JOIN(aggBrCum, splits, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOCAL);
      node_base := MAX(nodes, node_id);
      new_split := PROJECT(new_spl0, TRANSFORM(gNodeInstDisc, SELF.value:= node_base + COUNTER; SELF.depend:= LEFT.value;
                                               SELF.level:= p_level; SELF.support:= LEFT.TCnt; SELF := LEFT; SELF := [];));
      // reasigning instances to new nodes
      node_inst := JOIN(this_set, new_split, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.depend,
                      TRANSFORM(gNodeInstDisc, SELF.node_id:=RIGHT.value, SELF.level:= RIGHT.level +1, SELF.value:= LEFT.depend, SELF:= LEFT ), LOOKUP);
      RETURN pass_thru + new_split + node_inst;   // returning leaf nodes, new splits nodes and reassigned instances
    END;
    // generating best feature_selection-gini_impurity splits, loopfilter level = COUNTER let pass only the nodes to be splitted for any current level
    res   := LOOP(roots, LEFT.level=COUNTER, COUNTER < depth , RndFeatSelPartitionGRBased(ROWS(LEFT), COUNTER));
    // Turning LOOP results into splits and leaf nodes
    gSplitD toNewNode(gNodeInstDisc NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.value, 0);
      SELF.number       := IF(NodeInst.number>0, NodeInst.number, 0);
      SELF.value        := NodeInst.depend;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT));    // node splits and leaf nodes
    // Taking care of instances (id>0) that reached maximum level and did not turn into a leaf yet
    mode_r := RECORD
      res.group_id;
      res.node_id;
      res.level;
      res.depend;
      Cnt := COUNT(GROUP);
    END;
    depCnt := TABLE(res(id>0),mode_r, group_id, node_id, level, depend, FEW);
    maxlevel_leafs:= PROJECT(depCnt, TRANSFORM(gSplitD, SELF.number:=0, SELF.value:= LEFT.depend, SELF.support:= LEFT.cnt, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes + maxlevel_leafs;
  END;
  
  EXPORT SplitFeatureSampleIGR(DATASET(DiscreteField) Indep, DATASET(DiscreteField) Dep, t_Count treeNum, t_Count fsNum, t_level maxLevel=32) := FUNCTION
    N       := MAX(Dep, id);       // Number of Instances
    totFeat := COUNT(Indep(id=N)); // Number of Features
    depth   := MIN(255, maxLevel); // Max number of iterations when building trees (max 256 levels)
    dDep    := DISTRIBUTE(Dep   , HASH32(Id));
    dIndep  := DISTRIBUTE(Indep , HASH32(id));
    // sampling with replacement the original dataset to generate treeNum Datasets
    grList0   := Sampling.GenerateNSampleList(treeNum, N);  // mapping list between new group-instances and original instances
    dgrLstOld := DISTRIBUTE(grList0  , HASH32(oldId));      // oldId holds the correspondent original instance
    // for each tree assign all their sampled instances at root node
    roots     := JOIN(dDep  , dgrLstOld, LEFT.id = RIGHT.oldId, GenerateDiscRoots(LEFT, RIGHT), LOCAL);
    // Calculated only once, used at each iteration inside loopbody
    all_Data0 := JOIN(dIndep, dgrLstOld, LEFT.id = RIGHT.oldId, GetDRecords(LEFT, RIGHT) , LOCAL);
    all_Data  := DISTRIBUTE(all_Data0, HASH32(id));
    // loopbody function
    gNodeInstDisc RndFeatSelPartitionIGRBased(DATASET(gNodeInstDisc) nodes, t_Count p_level):= FUNCTION
      dNodes := DISTRIBUTE(nodes, HASH32(group_id, node_id));
      // Calculating Information Entropy of Nodes
      top_dep     := TABLE(dNodes , {group_id, node_id, depend, cnt:= COUNT(GROUP)}, group_id, node_id, depend, LOCAL);
      top_dep_tot := TABLE(top_dep, {group_id, node_id, tot:= SUM(GROUP, cnt)}     , group_id, node_id, LOCAL);
      tdp := RECORD
        top_dep;
        REAL4 prop; // Proportion based only on dependent value
        REAL4 plogp:= 0;
      END;
      P_Log_P(REAL P) := IF(P=1, 0, -P*LOG(P)/LOG(2));
      top_dep_p := JOIN(top_dep, top_dep_tot, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id,
                TRANSFORM(tdp, SELF.prop:= LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot), SELF:=LEFT), LOCAL);
      top_info  := TABLE(top_dep_p, {group_id, node_id, info:= SUM(GROUP, plogp)}, group_id, node_id, LOCAL); // Nodes Information Entropy
      // Filtering pure nodes
      PureNodes := top_info(info = 0);          // Pure Nodes have Info Entropy = 0
      NonPureNodes  := top_info(info > 0);      // Node-instances in pure nodes pass through
      pass_thru := JOIN(top_dep, PureNodes, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id, TRANSFORM(gNodeInstDisc,
                       SELF.level:= p_level, SELF.depend:=LEFT.depend, SELF.support:=LEFT.cnt, SELF.id:=0, SELF.number:=0, SELF.value:=0, SELF:=LEFT), LOCAL);
      // New working set after removing pass through node-instances
      nodes_toSplit := JOIN(dNodes, PureNodes, LEFT.node_id=RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOCAL);
      // Gather only the data needed for each LOOP iteration,
      NodexKoutofM(DATASET(RECORDOF(top_info)) topNodes, t_FieldNumber K, t_FieldNumber M) := FUNCTION  
        rndFeatRec:= RECORD
          t_Count       group_id  := 0;
          t_node        node_id   := 0;
          t_FieldNumber number    :=0;
          t_FieldReal   rnd       :=0;
        END;
        seed:= DATASET([{0,0,0,0}], rndFeatRec);
        node_seed := JOIN(topNodes, seed, TRUE, TRANSFORM(rndFeatRec, SELF:= LEFT), ALL); // one seed for each Node
        // generating K features per node map and sampling only K out of M per node
        allFields := NORMALIZE(node_seed, M, TRANSFORM(rndFeatRec, SELF.number:= (COUNTER % M) +1, SELF.rnd:=RANDOM(), SELF:=LEFT),LOCAL);
        allSorted := SORT(allFields, group_id, node_id, rnd, LOCAL);
        raw_set   := ENTH(allSorted, K, M, 1); // LOCAL doesn't give exact results
        RETURN TABLE(raw_set, {group_id, node_id, number});
      END;
      nodesFeatSet  := DISTRIBUTE(NodexKoutofM(NonPureNodes, fsNum, totFeat), HASH32(group_id, node_id));  // generating list of features selected for each node
      ftSetInst := JOIN(nodes_toSplit, nodesFeatSet, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id, TRANSFORM(gNodeInstDisc, SELF.number:= RIGHT.number, SELF:= LEFT), LOCAL);
      // Populating nodes' attributes to split
      toSplit   := JOIN(all_Data, DISTRIBUTE(ftSetInst, HASH32(id)), LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(gNodeInstDisc, SELF.value:= LEFT.value; SELF:= RIGHT;), LOCAL);
      this_set  := DISTRIBUTE(toSplit, HASH32(group_id, node_id));
      // Calculating Information Gain of possible splits
      child       := TABLE(this_set, {group_id, node_id, number, value, depend, cnt := COUNT(GROUP)}, group_id, node_id, number, value, depend, LOCAL);
      child_tot := TABLE(child,    {group_id, node_id, number, value, tot := SUM(GROUP, cnt)},      group_id, node_id, number, value, LOCAL);
      csp := RECORD
        child_tot;
        REAL4 prop;
        REAL4 plogp;
      END;
      // Calculating Intrinsic Information Entropy of each attribute(split) per node
      csplit_p  := JOIN(child_tot, top_dep_tot, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id,
                    TRANSFORM(csp, SELF.prop:= LEFT.tot/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.tot/RIGHT.tot), SELF:=LEFT), LOCAL);
      csplit    := TABLE(csplit_p, {group_id, node_id, number, split_info:=SUM(GROUP, plogp)}, group_id, node_id, number, LOCAL); // Intrinsic Info
      chp := RECORD
        child;
        REAL4 prop; // Proportion pertaining to this dependant value
        REAL4 plogp:= 0;
      END;
      // Information Entropy of new branches per split
      cprop     := JOIN(child, child_tot, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value,
                    TRANSFORM(chp, SELF.prop := LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot), SELF:=LEFT), LOCAL);
      cplogp    := TABLE(cprop, {group_id, node_id, number, value, cont:= SUM(GROUP,cnt), inf0:= SUM(GROUP, plogp)}, group_id, node_id, number, value, LOCAL);
      // Information Entropy of possible splits per node
      cinfo     := TABLE(cplogp, {group_id, node_id, number, info:=SUM(GROUP, cont*inf0)/SUM(GROUP, cont)}, group_id, node_id, number, LOCAL);
      gainRec := RECORD
        t_count group_id;
        t_node  node_id;
        t_Discrete number;
        REAL4 gain;
      END;
      // Information Gain of possible splits per node
      gain      := JOIN(cinfo, top_info, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id,
                    TRANSFORM(gainRec, SELF.gain:= RIGHT.info - LEFT.info, SELF:= LEFT), LOCAL);
      gainRateRec := RECORD
        t_count group_id;
        t_node node_id;
        t_Discrete number;
        REAL4 gain_ratio;
      END;
      // Information Gain Ratio of possible splits per node
      gainRatio := JOIN(gain, csplit, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number,
                      TRANSFORM(gainRateRec, SELF.gain_ratio:= LEFT.gain/RIGHT.split_info, SELF:= LEFT), LOCAL);
      // Selecting the split with max Info Gain Ratio per node
      split     := DEDUP(SORT(gainRatio, group_id, node_id, -gain_ratio, LOCAL), group_id, node_id, LOCAL);

      // new split nodes found
      new_spl0  := JOIN(child_tot, split, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOCAL);
      node_base := MAX(nodes, node_id);
      new_split := PROJECT(new_spl0, TRANSFORM(gNodeInstDisc, SELF.value:= node_base + COUNTER; SELF.depend:= LEFT.value;
                                               SELF.level:= p_level; SELF.support:= LEFT.tot; SELF := LEFT; SELF := [];));
      // reasigning instances to new nodes
      node_inst := JOIN(this_set, new_split, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.depend,
                      TRANSFORM(gNodeInstDisc, SELF.node_id:=RIGHT.value, SELF.level:= RIGHT.level +1, SELF.value:= LEFT.depend, SELF:= LEFT ), LOOKUP);
      RETURN pass_thru + new_split + node_inst;   // returning leaf nodes, new splits nodes and reassigned instances
    END;
    // generating best feature_selection-gini_impurity splits, loopfilter level = COUNTER let pass only the nodes to be splitted for any current level
    res := LOOP(roots, LEFT.level=COUNTER, COUNTER < depth , RndFeatSelPartitionIGRBased(ROWS(LEFT), COUNTER));
    // Turning LOOP results into splits and leaf nodes
    gSplitD toNewNode(gNodeInstDisc NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.value, 0);
      SELF.number       := IF(NodeInst.number>0, NodeInst.number, 0);
      SELF.value        := NodeInst.depend;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT));    // node splits and leaf nodes
    // Taking care of instances (id>0) that reached maximum level and did not turn into a leaf yet
    mode_r := RECORD
      res.group_id;
      res.node_id;
      res.level;
      res.depend;
      Cnt := COUNT(GROUP);
    END;
    depCnt := TABLE(res(id>0),mode_r, group_id, node_id, level, depend, FEW);
    maxlevel_leafs:= PROJECT(depCnt, TRANSFORM(gSplitD, SELF.number:=0, SELF.value:= LEFT.depend, SELF.support:= LEFT.cnt, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes + maxlevel_leafs;
  END;

  EXPORT FromDiscreteForest(DATASET(NumericField) mod) := FUNCTION
    FromField(mod, gSplitD,o, modelD_Map);
    RETURN o;
  END;
  EXPORT ToDiscreteForest(DATASET(gSplitD) nodes) := FUNCTION
    AppendID(nodes, id, model);
    ToField(model, out_model, id, modelD_fields);
    RETURN out_model;
  END;
  // Function that locates instances into the deepest branch nodes (split) based on their attribute values
  EXPORT gSplitInstancesD(DATASET(gSplitD) mod, DATASET(DiscreteField) Indep) := FUNCTION
    wNode := RECORD(gSplitD)
      REAL weight:= 1.0;
    END;
    inst_gnode:= RECORD(DiscreteField)
      t_Count group_id;
      NodeID;
      REAL weight:= 1.0;
    END;
    id_group := RECORD
      t_RecordID    id;
      t_Count group_id;
    END;
    depth:=MAX(mod, level);
    dMod := DISTRIBUTE(mod, HASH32(node_id, number));
    aCCmod := TABLE(dMod, {node_id, number, tot:= SUM(GROUP, support)}, node_id, number, LOCAL);
    wNodes := JOIN(dMod, aCCmod, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number, TRANSFORM(wNode,
                    SELF.weight:= LEFT.support/RIGHT.tot, SELF:=LEFT), LOCAL);
    ind0:= DISTRIBUTE(Indep, id);
    root:= mod(level = 1);        // This will contains one Node record per Tree, "100 trees" is a common value
    inst_root:= JOIN(ind0, root, LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value, TRANSFORM(inst_gnode, SELF:= LEFT, SELF:=RIGHT ), MANY LOOKUP);
    nTree:= MAX(root, group_id);
    pivot:= root[1].number;
    fake_root:=  NORMALIZE(ind0(number = pivot), nTree, TRANSFORM(id_group, SELF.id:= LEFT.id, SELF.group_id:= COUNTER), LOCAL);
    no_root := JOIN(fake_root, inst_root, LEFT.id = RIGHT.id AND LEFT.group_id = RIGHT.group_id,TRANSFORM(LEFT), LEFT ONLY, LOCAL) ;
    nr_inst := JOIN(no_root, root, LEFT.group_id = RIGHT.group_id, TRANSFORM(inst_gnode, SELF.id:= LEFT.id, SELF:= RIGHT), LOOKUP);
    inst_gnodes0:= inst_root + nr_inst;
    loop_body(DATASET(inst_gnode) inst_gnodes, UNSIGNED2 p_level) := FUNCTION
      nodesN:= wNodes(level=p_level);
      inst:= JOIN(inst_gnodes, ind0, LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number, TRANSFORM(inst_gnode, SELF.value:=RIGHT.value, SELF:= LEFT), LOCAL);
      join0:= JOIN(inst, nodesN, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value, LOOKUP, LEFT OUTER);
      miss_val:=JOIN(join0(new_node_id=0), nodesN, LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_gnode,
                    SELF.weight:= LEFT.weight*RIGHT.weight, SELF.level:=LEFT.level+1, SELF.node_id:=RIGHT.new_node_id, SELF:=LEFT), LOOKUP, MANY);
      match_val:=PROJECT(join0(new_node_id>0), TRANSFORM(inst_gnode, SELF.node_id:=LEFT.new_node_id, SELF.level:=LEFT.level+1, SELF:=LEFT), LOCAL);
      all_val:= miss_val + match_val;
      nxt_nodes:= wNodes(level=p_level +1);
      RETURN JOIN(all_val, nxt_nodes, LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_gnode, SELF.number:=RIGHT.number, SELF.value:= RIGHT.value, SELF:=LEFT), LOOKUP);
    END;
    RETURN LOOP(inst_gnodes0, depth, LEFT.number>0, loop_body(ROWS(LEFT), COUNTER));
  END;
  // Probability function for discrete independent values and model
  EXPORT ClassProbDistribForestD(DATASET(DiscreteField) Indep, DATASET(NumericField) mod) := FUNCTION
    nodes  := FromDiscreteForest(mod);
    dataSplitted:= gSplitInstancesD(nodes, Indep);
    dDS    := DISTRIBUTE(dataSplitted, HASH32(id));
    accDS  := TABLE(dDS, {id, group_id, value, sumWeight:= SUM(GROUP, weight)}, id, group_id, value, LOCAL);
    sortDS := SORT(accDS, id, group_id, value, -sumWeight, LOCAL);
    ddupDS := DEDUP(sortDS, id, group_id, LOCAL);
    accCDS := TABLE(ddupDS, {id, value, cnt:= COUNT(GROUP)}, id, value, LOCAL);
    tClass := TABLE(accCDS, {id, tot:= SUM(GROUP, cnt)}, id, LOCAL);
    sClass := JOIN(accCDS, tClass, LEFT.id=RIGHT.id, LOCAL);
    RETURN PROJECT(sClass, TRANSFORM(l_result, SELF.conf:= LEFT.cnt/LEFT.tot, SELF.number:= 1, SELF:= LEFT), LOCAL);
  END;
  // Classification function for discrete independent values and model
  EXPORT ClassifyDForest(DATASET(DiscreteField) Indep,DATASET(NumericField) mod) := FUNCTION
    // get class probabilities for each instance
    dClass:= ClassProbDistribForestD(Indep, mod);
    // select the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN PROJECT(finalClass, TRANSFORM(l_result, SELF:= LEFT, SELF:=[]), LOCAL);
  END;
  
/* Continuos implementation*/
// Function to binary-split a set of nodes based on Feature Selection and Gini Impurity,
// the nodes received were generated sampling with replacement nTrees times.
// Note: it selects kFeatSel out of mTotFeats features for each sample, features must start at 1 and cannot exist a gap in the numeration.  
  EXPORT RndFeatSelBinPartitionGIBased(DATASET(gNodeInstCont) nodes, t_Count nTrees, t_Count kFeatSel, t_Count mTotFeats, t_Count p_level, REAL Purity=1.0):= FUNCTION
    this_set_all := DISTRIBUTE(nodes, HASH(group_id, node_id, number));
    node_base := MAX(this_set_all, node_id);           // Start allocating new node-ids from the highest previous
    featSet:= NxKoutofM(nTrees, kFeatSel, mTotFeats);
    minFeats := TABLE(featSet, {gNum, minNumber := MIN(GROUP, number)}, gNum, FEW); // chose the min feature number from the sample
    this_minFeats:= JOIN(this_set_all, minFeats, LEFT.group_id = RIGHT.gNum AND LEFT.number= RIGHT.minNumber, LOOKUP);
    // Calculating dependent and total count for each node
    node_dep := TABLE(this_minFeats, {group_id, node_id, depend, cnt:= COUNT(GROUP)}, group_id, node_id, depend, FEW);
    node_dep_tot := TABLE(node_dep, {group_id, node_id, tot:= SUM(GROUP, cnt)}, group_id, node_id, FEW);
    r := RECORD
      node_dep;
      node_dep_tot.tot;
      REAL4 Prop; // Proportion pertaining to this dependant value
    END;
    node_prop := JOIN(node_dep, node_dep_tot,  LEFT.group_id = RIGHT.group_id AND LEFT.node_id =RIGHT.node_id,
                  TRANSFORM(r, SELF.Prop := LEFT.cnt/RIGHT.tot, SELF.tot:= RIGHT.tot, SELF := LEFT));
    // Compute 1-gini coefficient for each node for each field for each value
    gini_node:= TABLE(node_prop,{node_id, TotalCnt := SUM(GROUP,Cnt), Gini := 1-SUM(GROUP,Prop*Prop)}, node_id, FEW);
    PureEnough := gini_node(gini >= Purity);
    s_node_prop:= SORT(node_prop, group_id, node_id, -cnt);
    d_node_prop:= DEDUP(s_node_prop, group_id, node_id);
    leafsNodes := JOIN(d_node_prop, PureEnough, LEFT.node_id=RIGHT.node_id, TRANSFORM(gNodeInstCont, SELF.id:=0, SELF.number:=0, SELF.value:=0, SELF.level:= p_level,SELF:=LEFT), FEW);
    // splitting the instances that did not reach a leaf node
    this_set_out:= JOIN(this_set_all, PureEnough, LEFT.node_id=RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    this_set  := JOIN(this_set_out, featSet, LEFT.group_id = RIGHT.gNum AND LEFT.number= RIGHT.number, TRANSFORM(LEFT), LOOKUP);  
    ts_acc_dep   := TABLE(this_set, {group_id, node_id, number, value, depend, depcnt := COUNT(GROUP)}, group_id, node_id, number, value, depend, LOCAL);
    rec_dep:= RECORD
         ts_acc_dep;
         INTEGER tot_Low:=0;    // total number of ocurrences of Dependent with attrib-value <= treshold the Bag
         INTEGER tot_High:=0;   // total number of ocurrences of Dependent with attrib-value > treshold the Bag
         INTEGER tot_Dep:=0;    // total number of ocurrences of Dep value at the Bag
         INTEGER tot_Node:=0;   // total number of ocurrences at the Node
    END;
    rec_dep pop_dep(ts_acc_dep le, node_prop ri):= TRANSFORM
      SELF.depend:=   ri.depend;
      SELF.depcnt:=   IF(le.depend= ri.depend, le.depcnt, 0);
      SELF.tot_Dep:=  ri.cnt;
      SELF.tot_Node:= ri.tot;
      SELF:=          le;
      SELF:=          ri;
    END;
    deps:= JOIN(ts_acc_dep, node_prop, LEFT.node_id = RIGHT.node_id, pop_dep(LEFT, RIGHT), MANY LOOKUP);
    sort_deps:= SORT(deps, group_id, node_id, number, value, depend, -depcnt, LOCAL);
    ddup_deps:= DEDUP(sort_deps, group_id, node_id, number, depend, value, LOCAL);
    dist_deps:= DISTRIBUTE(ddup_deps, HASH(node_id, number, depend), MERGE(node_id, number, depend, value));
    rec_dep rold(dist_deps le, dist_deps ri) := TRANSFORM
      SELF.tot_Low:= ri.depCnt + IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend , le.tot_Low, 0);
      SELF.tot_High:= ri.tot_dep - ri.depCnt - IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend, le.tot_Low, 0);
      SELF := ri;
    END;
    // Accumulated Counting per Dependent value per Cut threshold
    bag_grouped := ITERATE(dist_deps, rold(LEFT,RIGHT), LOCAL);
    sp_bag:= TABLE(bag_grouped, {group_id, node_id, number, value, acc_low:= SUM(GROUP, tot_low), acc_high:= SUM(GROUP, tot_high)}, group_id, node_id, number, value);
    b_prop:= RECORD
      bag_grouped.group_id;
      bag_grouped.node_id;
      bag_grouped.number;
      bag_grouped.value;
      bag_grouped.depend;
      bag_grouped.tot_low;
      sp_bag.acc_low;
      REAL4 dep_prop_low;
      bag_grouped.tot_high;
      sp_bag.acc_high;
      REAL4 dep_prop_high;
      bag_grouped.tot_node;
    END;
    bag_prop:= JOIN(bag_grouped, sp_bag, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id
              AND LEFT.number = RIGHT.number AND LEFT.value = RIGHT.value,
              TRANSFORM(b_prop, SELF.dep_prop_low := LEFT.tot_low/RIGHT.acc_low,
                                SELF.dep_prop_high := LEFT.tot_high/RIGHT.acc_high,
                                SELF := RIGHT, SELF:=LEFT), HASH);
    zigma:= TABLE(bag_prop, {group_id, node_id, number, value,
                    propL:= acc_low/tot_node, giniL:= 1-SUM(GROUP,dep_prop_low*dep_prop_low),
                    propH:= acc_High/tot_node, giniH:= 1-SUM(GROUP,dep_prop_high*dep_prop_high)},
                    group_id, node_id, number, value, LOCAL);
    gini_rec:= RECORD
      zigma.group_id;
      zigma.node_id;
      zigma.number;
      zigma.value;
      REAL4 gini_t:=0;
    END;
    sp_gini:= PROJECT(zigma, TRANSFORM(gini_rec, SELF.gini_t:= LEFT.propL*LEFT.giniL + LEFT.propH*LEFT.giniH, SELF:=LEFT), LOCAL);
    sort_sp:= SORT(sp_gini, group_id, node_id);
    sort_gini:= SORT(sort_sp, group_id, node_id, gini_t, LOCAL);
    node_splits:= DEDUP(sort_gini, group_id, node_id, LOCAL);
    // Start allocating new node-ids from the highest previous
    new_nodes_low:= PROJECT(node_splits, TRANSFORM(gNodeInstCont, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER -1, SELF.level:= p_level, SELF.high_fork:=FALSE, SELF := LEFT));
    new_nodes_high:= PROJECT(node_splits, TRANSFORM(gNodeInstCont, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER, SELF.level:= p_level, SELF.high_fork:=TRUE, SELF := LEFT));
    new_nodes:= new_nodes_low + new_nodes_high;
    // Assignig instances that didn't reach a leaf node to (new) node-ids (by joining to the sampled data)
    noleaf:= JOIN(this_set_out, leafsNodes, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id, LEFT ONLY, LOOKUP);
    r1 := RECORD
      t_Recordid id;
      t_node nodeid;
      BOOLEAN high_fork:=FALSE;
    END;
    mapp := JOIN(noleaf, new_nodes, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND (LEFT.value>RIGHT.value)= RIGHT.high_fork,
                TRANSFORM(r1, SELF.id := LEFT.id, SELF.nodeid:=RIGHT.depend, SELF.high_fork:=RIGHT.high_fork ),LOOKUP);
    // Now use the mapping to actually reset all the points
    J := JOIN(this_set_out, mapp, LEFT.id=RIGHT.id, TRANSFORM(gNodeInstCont, SELF.node_id:=RIGHT.nodeid, SELF.level:=LEFT.level+1, SELF := LEFT), LOOKUP);
    RETURN nodes(level < p_level) + leafsNodes + new_nodes + J;
  END;
// Function used in Random Forest Classifier Continuos Learning
// Note: returns treeNum Binary Decision Trees, split based on Gini Impurity
//       it selects fsNum out of total number of features, they must start at 1 and cannot exist a gap in the numeration.
//       Gini Impurity's default parameters: Purity = 1.0 and maxLevel (Depth) = 32 (up to 126 max iterations)
  EXPORT SplitFeatureSampleGIBin(DATASET(NumericField) Indep, DATASET(DiscreteField) Dep, t_Count treeNum, t_Count fsNum, REAL Purity=1.0, t_level maxLevel=32) := FUNCTION
    N       := MAX(Dep, id);       // Number of Instances
    totFeat := COUNT(Indep(id=N)); // Number of Features
    depth   := MIN(126, maxLevel); // Max number of iterations when building trees (max 126 levels)
    // sampling with replacement the original dataset to generate treeNum Datasets
    grList:= ML.Sampling.GenerateNSampleList(treeNum, N); // the number of records will be N * treeNum
    groupDep0:= JOIN(dep, grList, LEFT.id = RIGHT.oldId, GroupDepRecords(LEFT, RIGHT));
    groupDep:=DISTRIBUTE(groupDep0, HASH(id));
    ind0 := ML.Utils.Fat(Indep); // Ensure no sparsity in independents
    gNodeInstCont init(NumericField ind, DepGroupedRec depG) := TRANSFORM
      SELF.group_id := depG.group_id;
      SELF.node_id := depG.group_id;
      SELF.level := 1;
      SELF.depend := depG.value;	// Actually copies the dependant value to EVERY node - paying memory to avoid downstream cycles
      SELF.id := depG.new_id;
      SELF := ind;
    END;
    ind1 := JOIN(ind0, groupDep, LEFT.id = RIGHT.id, init(LEFT,RIGHT), LOCAL); 
    // generating best feature_selection-gini_impurity splits, loopfilter level = COUNTER let pass only the nodes to be splitted for any current level
    res := LOOP(ind1,  LEFT.level=COUNTER AND LEFT.level<= depth, RndFeatSelBinPartitionGIBased(ROWS(LEFT), treeNum, fsNum, totFeat, COUNTER, Purity));
    // Turning LOOP results into splits and leaf nodes
    gSplitC toNewNode(gNodeInstCont NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.depend, 0);
      SELF.value := IF(NodeInst.number>0, NodeInst.value, NodeInst.depend);
      SELF.high_fork:=(INTEGER1)NodeInst.high_fork;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT), LOCAL);    // node splits and leaf nodes
    mode_r := RECORD
      res.group_id;
      res.node_id;
      res.level;
      res.depend;
      cnt := COUNT(GROUP);
    END;
    // Taking care instances (id>0) that reached maximum level and did not turn into a leaf yet
    depCnt      := TABLE(res(id>0, number=1), mode_r, group_id, node_id, level, depend, FEW);
    // Assigning class value based on majority voting
    depCntSort  := SORT(depCnt, group_id, node_id, -cnt); // if more than one dependent value for node_id
    depCntDedup := DEDUP(depCntSort, group_id, node_id);     // the class value with more counts is selected
    maxlevel_leafs:= PROJECT(depCntDedup, TRANSFORM(gSplitC, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes + maxlevel_leafs;
  END;
  
  EXPORT SplitFeatureSampleGIBin2(DATASET(NumericField) Indep, DATASET(DiscreteField) Dep, t_Count treeNum, t_Count fsNum, REAL Purity=1.0, t_level maxLevel=32) := FUNCTION
    N       := MAX(Dep, id);       // Number of Instances
    totFeat := COUNT(Indep(id=N)); // Number of Features
    depth   := MIN(255, maxLevel); // Max number of iterations when building trees (max 256 levels)
    dDep    := DISTRIBUTE(Dep   , HASH32(Id));
    dIndep  := DISTRIBUTE(Indep , HASH32(id));
    // sampling with replacement the original dataset to generate treeNum Datasets
    grList0   := Sampling.GenerateNSampleList(treeNum, N);  // mapping list between new group-instances and original instances
    dgrLstOld := DISTRIBUTE(grList0  , HASH32(oldId));      // oldId holds the correspondent original instance
    // for each tree assign all their sampled instances at root node
    roots     := JOIN(dDep  , dgrLstOld, LEFT.id = RIGHT.oldId, GenerateContRoots(LEFT, RIGHT), LOCAL);
    // Calculated only once, used at each iteration inside loopbody
    all_Data0 := JOIN(dIndep, dgrLstOld, LEFT.id = RIGHT.oldId, GetCRecords(LEFT, RIGHT) , LOCAL);
    all_Data  := DISTRIBUTE(all_Data0, HASH32(id));
    // loopbody function
    gNodeInstCont RndFeatSelBinPartitionGIBased(DATASET(gNodeInstCont) nodes, t_Count p_level):= FUNCTION
      dNodes := DISTRIBUTE(nodes, HASH32(group_id, node_id));
      // Calculating Gini Impurity for each node
      aggNode  := TABLE(dNodes,  {group_id, node_id, depend, cnt := COUNT(GROUP)}, group_id, node_id, depend, LOCAL);
      aggNodec := TABLE(aggNode, {group_id, node_id,     tcnt := SUM(GROUP, cnt)}, group_id, node_id,         LOCAL);
      recNode  := RECORD
        aggNode;
        aggNodec.tcnt;
        REAL4 prop; // Proportion pertaining to its dependant value
      END;
      propNode   := JOIN(aggNode, aggNodec, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id, TRANSFORM(recNode, SELF.prop := LEFT.cnt/RIGHT.tcnt, SELF.tcnt:=RIGHT.tcnt, SELF := LEFT), LOCAL);
      Purities   := TABLE(propNode, {group_id, node_id, totalCnt := SUM(GROUP, cnt), gini := 1-SUM(GROUP, prop*prop)}, group_id, node_id, LOCAL); // Compute gini purity for each node   
      // Filtering pure and non-pure nodes
      PureEnough := Purities(1-Purity >= gini); // Pure nodes filter
      NotPure    := Purities(1-Purity <  gini); // Non pure nodes filter
      isPureNode := JOIN(dNodes, PureEnough, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id, 
                            TRANSFORM({gNodeInstCont, BOOLEAN pure:= FALSE}, SELF.pure:=RIGHT.node_id>0, SELF:=LEFT), LEFT OUTER, LOCAL);
      pass_thru    := PROJECT(isPureNode(pure = TRUE), gNodeInstCont, LOCAL);
      ImpureNodes  := PROJECT(isPureNode(pure = FALSE), gNodeInstCont, LOCAL); // still distributted by group_id, node_id
      // Gather only the data needed for each LOOP iteration,
      NodexKoutofM(DATASET(RECORDOF(Purities)) topNodes, t_FieldNumber K, t_FieldNumber M) := FUNCTION  
        rndFeatRec:= RECORD
          t_Count       group_id  := 0;
          t_node        node_id   := 0;
          t_FieldNumber number    := 0;
          t_FieldReal   rnd       := 0;
        END;
        seed:= DATASET([{0,0,0,0}], rndFeatRec);
        node_seed := JOIN(topNodes, seed, TRUE, TRANSFORM(rndFeatRec, SELF:= LEFT), ALL); // one seed for each Node
        // generating K features per node map and sampling only K out of M per node
        allFields := NORMALIZE(node_seed, M, TRANSFORM(rndFeatRec, SELF.number:= (COUNTER % M) +1, SELF.rnd:=RANDOM(), SELF:=LEFT),LOCAL);
        allSorted := SORT(allFields, group_id, node_id, rnd, LOCAL);
        raw_set   := ENTH(allSorted, K, M, 1); // LOCAL doesn't give exact results
        RETURN TABLE(raw_set, {group_id, node_id, number});
      END;
      nodesFeatSet  := DISTRIBUTE(NodexKoutofM(NotPure, fsNum, totFeat), HASH32(group_id, node_id));  // generating list of features selected for each node and distributing to match NonPureNodes
      // Populating nodes' attributes to split
      ftSetInst := JOIN(ImpureNodes, nodesFeatSet, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id, TRANSFORM(gNodeInstCont, SELF.number:= RIGHT.number, SELF:= LEFT), LOCAL);
      toSplit   := JOIN(all_Data, DISTRIBUTE(ftSetInst, HASH32(id)), LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number, TRANSFORM(gNodeInstCont, SELF.value:= LEFT.value; SELF:= RIGHT;), LOCAL);
      // new set of nodes-instance-att_value to find best split per node based on gini impurity
      this_set  := DISTRIBUTE(toSplit, HASH32(group_id, node_id));
      acc_dep   := TABLE(this_set, {group_id, node_id, number, value, depend, depcnt := COUNT(GROUP)}, group_id, node_id, number, value, depend, LOCAL);
      rec_dep:= RECORD
           acc_dep;
           INTEGER tot_Low:=0;    // total number of ocurrences of Dependent with attrib-value <= treshold the Bag
           INTEGER tot_High:=0;   // total number of ocurrences of Dependent with attrib-value > treshold the Bag
           INTEGER tot_Dep:=0;    // total number of ocurrences of Dep value at the Bag
           INTEGER tot_Node:=0;   // total number of ocurrences at the Node
      END;
      rec_dep pop_dep(acc_dep le, propNode ri):= TRANSFORM
        SELF.depend:=   ri.depend;
        SELF.depcnt:=   IF(le.depend= ri.depend, le.depcnt, 0);
        SELF.tot_Dep:=  ri.cnt;
        SELF.tot_Node:= ri.tcnt;
        SELF:=          le;
        SELF:=          ri;
      END;
      deps      := JOIN(acc_dep, propNode, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id, pop_dep(LEFT, RIGHT), LOCAL);
      sort_deps := SORT(deps,       group_id, node_id, number, value, depend, -depcnt, LOCAL);
      ddup_deps := DEDUP(sort_deps, group_id, node_id, number, depend, value, LOCAL);
      dist_deps := DISTRIBUTE(ddup_deps, HASH(group_id, node_id, number, depend), MERGE(group_id, node_id, number, depend, value));
      rec_dep rold(dist_deps le, dist_deps ri) := TRANSFORM
        SELF.tot_Low  :=              ri.depCnt + IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend, le.tot_Low, 0);
        SELF.tot_High := ri.tot_dep - ri.depCnt - IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend, le.tot_Low, 0);
        SELF := ri;
      END;
      // Accumulated Counting per Dependent value per Cut threshold
      bag_grouped := ITERATE(dist_deps, rold(LEFT,RIGHT), LOCAL);
      sp_bag:= TABLE(bag_grouped, {group_id, node_id, number, value, acc_low:= SUM(GROUP, tot_low), acc_high:= SUM(GROUP, tot_high)}, group_id, node_id, number, value);
      b_prop:= RECORD
        bag_grouped.group_id;
        bag_grouped.node_id;
        bag_grouped.number;
        bag_grouped.value;
        bag_grouped.depend;
        bag_grouped.tot_low;
        sp_bag.acc_low;
        REAL4 dep_prop_low;
        bag_grouped.tot_high;
        sp_bag.acc_high;
        REAL4 dep_prop_high;
        bag_grouped.tot_node;
      END;
      bag_prop:= JOIN(bag_grouped, sp_bag, LEFT.group_id = RIGHT.group_id AND LEFT.node_id = RIGHT.node_id
                AND LEFT.number = RIGHT.number AND LEFT.value = RIGHT.value,
                TRANSFORM(b_prop, SELF.dep_prop_low := LEFT.tot_low/RIGHT.acc_low,
                                  SELF.dep_prop_high := LEFT.tot_high/RIGHT.acc_high,
                                  SELF := RIGHT, SELF:=LEFT), HASH);
      zigma:= TABLE(bag_prop, {group_id, node_id, number, value,
                      propL:= acc_low/tot_node, giniL:= 1-SUM(GROUP,dep_prop_low*dep_prop_low),
                      propH:= acc_High/tot_node, giniH:= 1-SUM(GROUP,dep_prop_high*dep_prop_high)},
                      group_id, node_id, number, value, LOCAL);      
      gini_rec:= RECORD
        zigma.group_id;
        zigma.node_id;
        zigma.number;
        zigma.value;
        REAL4 gini_t:=0;
      END;
      sp_gini:= PROJECT(zigma, TRANSFORM(gini_rec, SELF.gini_t:= LEFT.propL*LEFT.giniL + LEFT.propH*LEFT.giniH, SELF:=LEFT), LOCAL);
      sort_sp:= SORT(sp_gini, group_id, node_id);
      sort_gini:= SORT(sort_sp, group_id, node_id, gini_t, LOCAL);
      node_splits:= DEDUP(sort_gini, group_id, node_id, LOCAL);
      // Start allocating new node-ids from the highest previous
      node_base := MAX(nodes, node_id);
      new_nodes_low  := PROJECT(node_splits, TRANSFORM(gNodeInstCont, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER -1, SELF.level:= p_level, SELF.high_fork:=FALSE, SELF := LEFT));
      new_nodes_high := PROJECT(node_splits, TRANSFORM(gNodeInstCont, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER   , SELF.level:= p_level, SELF.high_fork:=TRUE , SELF := LEFT));
      new_nodes:= new_nodes_low + new_nodes_high;
      node_inst := JOIN(this_set, new_nodes, LEFT.group_id = RIGHT.group_id AND LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND (LEFT.value>RIGHT.value)= RIGHT.high_fork,
                      TRANSFORM(gNodeInstCont, SELF.node_id:=RIGHT.depend, SELF.level:= RIGHT.level +1, SELF:= LEFT ), LOOKUP);
    // mapp := JOIN(noleaf, new_nodes, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND (LEFT.value>RIGHT.value)= RIGHT.high_fork,
                // TRANSFORM(r1, SELF.id := LEFT.id, SELF.nodeid:=RIGHT.depend, SELF.high_fork:=RIGHT.high_fork ),LOOKUP);    
      RETURN pass_thru + new_nodes + node_inst;   // returning leaf nodes, new splits nodes and reassigned instances
    END;
    // generating best feature_selection-gini_impurity splits, loopfilter level = COUNTER let pass only the nodes to be splitted for any current level
    res := LOOP(roots,  LEFT.level=COUNTER AND LEFT.level<= depth, RndFeatSelBinPartitionGIBased(ROWS(LEFT), COUNTER));
    // Turning LOOP results into splits and leaf nodes
    gSplitC toNewNode(gNodeInstCont NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.depend, 0);
      SELF.value := IF(NodeInst.number>0, NodeInst.value, NodeInst.depend);
      SELF.high_fork:=(INTEGER1)NodeInst.high_fork;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT), LOCAL);    // node splits
    mode_r := RECORD
      res.group_id;
      res.node_id;
      res.level;
      res.depend;
      cnt := COUNT(GROUP);
    END;
    // Fitting pass_thru and max-level node-instances (id>0) into leaf nodes. Assigning class(depend) value based on majority voting.
    depCnt      := TABLE(res(id>0), mode_r, group_id, node_id, level, depend, FEW);
    depCntSort  := SORT(depCnt, group_id, node_id, -cnt); // if more than one dependent value for node_id
    depCntDedup := DEDUP(depCntSort, group_id, node_id);     // the class value with more counts is selected
    to_leafs:= PROJECT(depCntDedup, TRANSFORM(gSplitC, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes + to_leafs;
  END;  
  
  EXPORT ToContinuosForest(DATASET(gSplitC) nodes) := FUNCTION
    AppendID(nodes, id, model);
    ToField(model, out_model, id, modelC_fields);
    RETURN out_model;
  END;
  EXPORT FromContinuosForest(DATASET(NumericField) mod) := FUNCTION
    ML.FromField(mod, gSplitC,o, modelC_Map);
    RETURN o;
  END;
  // Function that locates instances into the deepest branch nodes (split) based on their attribute values
  EXPORT gSplitInstC(DATASET(gSplitC) mod, DATASET(NumericField) Indep) := FUNCTION
    splits:= mod(new_node_id <> 0);	// separate split or branches
    leafs := mod(new_node_id = 0);	// from final nodes
    Ind   := DISTRIBUTE(Indep, HASH(id));
    join0 := JOIN(Ind, splits, LEFT.number = RIGHT.number AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), LOOKUP, MANY);
    sort0 := SORT(join0, group_id, id, level, node_id, LOCAL);
    dedup0:= DEDUP(sort0, LEFT.group_id = RIGHT.group_id AND LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT, LOCAL);
    RETURN DEDUP(dedup0, LEFT.group_id = RIGHT.group_id AND LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT, LOCAL);
  END;
  // Function that locates instances into the deepest branch nodes (split) based on their attribute values
  // EXPORT gSplitInstancesC(DATASET(gSplitC) mod, DATASET(NumericField) Indep) := FUNCTION
    // inst_gnode:= RECORD(NumericField)
      // t_Count  group_id;
      // NodeID;
      // INTEGER1 high_fork:=0;
      // t_node   new_node_id; 
    // END;
    // depth  := MAX(mod, level);
    // dNodes := DISTRIBUTE(mod, HASH32(node_id, number));
    // ind0   := DISTRIBUTE(Indep, id);
    // root   := mod(level = 1);        // This will contains one Node record per Tree, "100 trees" is a common value
    // inst_root:= JOIN(ind0, root, LEFT.number=RIGHT.number AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), TRANSFORM(inst_gnode, SELF:= LEFT, SELF:=RIGHT ), MANY LOOKUP);
    // loop_body(DATASET(inst_gnode) inst_gnodes, UNSIGNED2 p_level) := FUNCTION
      // nxt_nodes  := dNodes(level=p_level +1);
      // instNewNum := JOIN(inst_gnodes, nxt_nodes(high_fork=0), LEFT.new_node_id=RIGHT.node_id, TRANSFORM(inst_gnode, SELF.number:=RIGHT.number, SELF:= LEFT), LOOKUP);
      // pass_thru  := instNewNum(number=0);
      // instNewVal := JOIN(ind0, instNewNum(number>0), LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number, TRANSFORM(inst_gnode, SELF.value:=LEFT.value, SELF:=RIGHT), LOCAL);
      // instNewNod := JOIN(instNewVal, nxt_nodes, LEFT.new_node_id=RIGHT.node_id AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), TRANSFORM(inst_gnode, SELF:=RIGHT, SELF:=LEFT), LOOKUP);
      // RETURN pass_thru + instNewNod;
    // END;
    // RETURN LOOP(inst_root, depth, LEFT.number>0, loop_body(ROWS(LEFT), COUNTER));
  // END;
  EXPORT gSplitInstancesC(DATASET(gSplitC) mod, DATASET(NumericField) Indep) := FUNCTION
    inst_gnode:= RECORD(NumericField)
      t_Count  group_id;
      NodeID;
      INTEGER1 high_fork:=0;
      t_node   new_node_id; 
    END;
    depth  := MAX(mod, level);
    dNodes := DISTRIBUTE(mod, HASH32(node_id, number));
    ind0   := DISTRIBUTE(Indep, id);
    minAtt := MIN(ind0, number);
    root_low   := dNodes(level = 1, high_fork=0);        // This will contains one Node record per Tree, "100 trees" is a common value
    inst_root:= JOIN(ind0(number= minAtt), root_low, TRUE, TRANSFORM(inst_gnode, SELF:= LEFT, SELF.level:=0, SELF.new_node_id:= RIGHT.node_id, SELF:=RIGHT), ALL);
    loop_body(DATASET(inst_gnode) inst_gnodes, UNSIGNED2 p_level) := FUNCTION
      nxt_nodes  := dNodes(level=p_level);
      instNewNum := JOIN(inst_gnodes, nxt_nodes(high_fork=0), LEFT.new_node_id=RIGHT.node_id, TRANSFORM(inst_gnode, SELF.number:=RIGHT.number, SELF:= LEFT), LOOKUP);
      pass_thru  := instNewNum(number=0);
      instNewVal := JOIN(ind0, instNewNum(number>0), LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number, TRANSFORM(inst_gnode, SELF.value:=LEFT.value, SELF:=RIGHT), LOCAL);
      instNewNod := JOIN(instNewVal, nxt_nodes, LEFT.new_node_id=RIGHT.node_id AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), TRANSFORM(inst_gnode, SELF:=RIGHT, SELF:=LEFT), LOOKUP);
      RETURN pass_thru + instNewNod;
    END;
    RETURN LOOP(inst_root, depth, LEFT.number>0, loop_body(ROWS(LEFT), COUNTER));
  END;
  // Probability function for continuous independent values and model
  EXPORT ClassProbDistribForestC(DATASET(NumericField) Indep, DATASET(NumericField) mod) := FUNCTION
    nodes := FromContinuosForest(mod);
    leafs := nodes(new_node_id = 0);	// from final nodes
    splitData_raw:= gSplitInstancesC(nodes, Indep);
    splitData:= DISTRIBUTE(splitData_raw, id);
    gClass:= JOIN(splitData, leafs, LEFT.new_node_id = RIGHT.node_id AND LEFT.group_id = RIGHT.group_id,
              TRANSFORM(DiscreteField, SELF.id:= LEFT.id, SELF.number := 1, SELF.value:= RIGHT.value), LOOKUP);
    accClass:= TABLE(gClass, {id, number, value, cnt:= COUNT(GROUP)}, id, number, value, LOCAL);
    tClass := TABLE(accClass, {id, number, tot:= SUM(GROUP, cnt)}, id, number, LOCAL);
    sClass:= JOIN(accClass, tClass, LEFT.number=RIGHT.number AND LEFT.id=RIGHT.id, LOCAL);
    RETURN PROJECT(sClass, TRANSFORM(l_result, SELF.conf:= LEFT.cnt/LEFT.tot, SELF:= LEFT, SELF:=[]), LOCAL);
  END;
  // Classification function for continuous independent values and model
  EXPORT ClassifyCForest(DATASET(NumericField) Indep,DATASET(NumericField) mod) := FUNCTION
    // get class probabilities for each instance
    dClass:= ClassProbDistribForestC(Indep, mod);
    // select the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN PROJECT(finalClass, TRANSFORM(l_result, SELF:= LEFT, SELF:=[]), LOCAL);
  END;
END;