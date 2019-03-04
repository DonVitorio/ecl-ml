IMPORT ML;
IMPORT ML.Mat;

EXPORT Trees := MODULE
  EXPORT model_Map :=	DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},{'value','4'},{'new_node_id','5'},{'support','6'}], {STRING orig_name; STRING assigned_name;});
  EXPORT STRING model_fields := 'node_id,level,number,value,new_node_id,support';	// need to use field map to call FromField later
  EXPORT modelC_Map :=	DATASET([{'id','ID'},{'node_id','1'},{'level','2'},{'number','3'},{'value','4'},{'high_fork','5'},{'new_node_id','6'}], {STRING orig_name; STRING assigned_name;});
  EXPORT STRING modelC_fields := 'node_id,level,number,value,high_fork,new_node_id';	// need to use field map to call FromField later
  EXPORT Node := RECORD
    ML.Types.NodeID;
    ML.Types.NumericField;
  END;
  EXPORT sNode:= RECORD
    ML.Types.t_RecordID splitId:=0;
    Node;
    BOOLEAN HighBranch:= FALSE;
  END;
  EXPORT NodeInstDiscrete := RECORD
    ML.Types.NodeID;
    ML.Types.DiscreteField;      // Discrete Instance Data
    ML.Types.t_Discrete depend;  // Instance's dependant Value
    ML.Types.t_Count support:=0; // Support during learning
  END;
  EXPORT cNode := RECORD
    ML.Types.NodeID;
    ML.Types.t_Discrete depend; // The dependant value
    ML.Types.NumericField;
    BOOLEAN high_fork:=FALSE;
  END;
  EXPORT SplitD := RECORD		// data structure for splitting results
    ML.Types.NodeID;
    ML.Types.t_FieldNumber number;       // The attribute used to split
    ML.Types.t_Discrete    value;        // The discrete value for the attribute in question
    ML.Types.t_node        new_node_id;  // The new node identifier this branch links to
    ML.Types.t_Count       support:=0;   // Support during learning
  END;
  EXPORT SplitF := SplitD;
  EXPORT SplitC := RECORD
    ML.Types.NodeID;
    ML.Types.t_FieldNumber number;       // The attribute used to split
    ML.Types.t_FieldReal   value;        // The cutpoint value for the attribute in question
    INTEGER1               high_fork:=0; // Fork Flag - 0: lower or equal than value, 1: greater than value
    ML.Types.t_node        new_node_id;  // The new node identifier this branch links to
  END;
  // Learning Data Structures - Internal
  SHARED final_node := RECORD
    ML.Types.t_RecordID root_node   := 0; // parent node id
    ML.Types.t_level    root_level  := 0; // parent node level
    ML.Types.t_RecordID final_node  := 0; // final node id, '0' means the parent node is a leaf
    ML.Types.t_level    final_level := 0; // final node level
    ML.Types.t_Discrete final_class :=-1; // final class value, '-1' means the parent node is a branch
  END;
	SHARED final_node_instance := RECORD(final_node)
    ML.Types.t_RecordID instance_id		:= 0; 	// instance id
    ML.Types.t_Discrete instance_class	:= -1;	// instance class value
    BOOLEAN match:= FALSE;
  END;
  SHARED node_error := RECORD(final_node)
    UNSIGNED4 			e:=0;		// error count
    UNSIGNED4 		cnt:=0;		// total count
    REAL8 	NxErr_est:=0;		// N x error estimated
  END;
  // Learning TRANSFORMs and FUNCTIONs - Internal
	SHARED node_error to_node_error(SplitF l):= TRANSFORM
		SELF.root_node		:= l.node_id;
		SELF.root_level		:= l.level;
		SELF.final_node		:= l.new_node_id;
		SELF.final_level	:= l.level + 1;
	END;
/*
	The NodeIds within a KdTree follow a natural pattern - all the node-ids will have the same number of bits - corresponding to the
  depth of the tree+1. The left-most will always be 1. Moving from left to right a 0 always implies taking the 'low' decision at a node
  and a 1 corresponds to taking a 'high'. Thus an ID of 6 = 110 has been split twice; and this group is in the high then low group
  The Splits show the number and value used to split at each point
*/
  EXPORT KdTree(DATASET(ML.Types.NumericField) f,ML.Types.t_level Depth=10,ML.Types.t_level MedianDepth=0) := MODULE
	// Cannot presently support median computation on more than 32K nodes at once due to use of FieldAggregate library
	MedDepth := MIN(MedianDepth,15);
	  // Each iteration attempts to work with the next level down the tree; resolving multiple sub-trees at once
		// The reason is to ensure that the full cluster is busy all the time
		// It is assumed that all of the data-nodes are distributed by HASH(id) throughout
  Split(DATASET(Node) nodes, ML.Types.t_level p_level) := FUNCTION
    working_nodes:=nodes(level=p_level);
/*
    // For every node_id this computes the maximum and minimum extent of the space
    spans := TABLE(working_nodes,{ minv := MIN(GROUP,value); maxv := MAX(GROUP,value); cnt := COUNT(GROUP); node_id,number }, node_id,number, MERGE);
*/
    // For every node_id this computes the maximum and minimum extent of the space and variance
    spans := TABLE(working_nodes,{ minv := MIN(GROUP,value); maxv := MAX(GROUP,value); var:= VARIANCE(GROUP,value); cnt := COUNT(GROUP); node_id,number }, node_id,number, MERGE);
    leafspans:= spans(cnt=1);
    onlyvalspan:= spans(cnt>1 and maxv = minv);
    splitwannabes:= spans(cnt>1 and maxv > minv);
/*
    // Now find the split points - that is the number with the largest span for each node_id, excluding leafs and only one value spans
    sp := DEDUP( SORT( DISTRIBUTE(splitwannabes, HASH(node_id)),node_id,minv-maxv,LOCAL), node_id, LOCAL );// Here we compute the split point based upon the mean of the range
*/
    // Now find the split points - that is the number with the largest variance (more balanced tree) for each node_id, excluding leafs and only one value spans
    sp := DEDUP( SORT( DISTRIBUTE(splitwannabes, HASH(node_id)),node_id, -var,LOCAL), node_id, LOCAL );// Here we compute the split point based upon the variance
    pass:= JOIN(onlyvalspan, sp, LEFT.node_id = RIGHT.node_id, LEFT ONLY, LOOKUP);
    // Here we compute the split point based upon the mean of the range
    splits_mean := PROJECT( sp, TRANSFORM(Node,SELF.Id := 0, SELF.level := p_level, SELF.value := (LEFT.maxv+LEFT.minv)/2, SELF := LEFT));
		// Here we create split points based upon the median
		// this gives even split points - but it adds an NLgN process into the loop ...
		// Method currently uses field aggregates - which requires the node-id to fit into 16 bits
    into_med := JOIN(working_nodes,sp,LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number,TRANSFORM(ML.Types.NumericField,SELF.Number := LEFT.node_id,SELF := LEFT),LOOKUP);
    // Transform into splits format - but oops - field is missing
    // When median = minval we use nextval instead of median to avoid endless right-node propagation (all points >= than split value)
    s_median := PROJECT( ML.FieldAggregates(into_med).minMedNext, TRANSFORM(Node, SELF.Id := 0, SELF.level := p_level, SELF.node_id:=LEFT.number, SELF.value := IF(LEFT.median = LEFT.minval, LEFT.nextval, LEFT.median), SELF.number := 0));
    splits_median := JOIN(s_median,sp,LEFT.node_id=RIGHT.node_id,TRANSFORM(Node,SELF.number := RIGHT.number,SELF := LEFT),FEW);
    splits := IF ( p_level < MedDepth, splits_median, splits_mean );
		// based upon the split points we now partition the data - note the split information is assumed to fit inside RAM
		// First we perform the split on field to get record_id/node_id pairs
		r := RECORD
		  ML.Types.t_RecordId id;
			ML.Types.t_node node_id;
		END;
		r NoteNI(working_nodes le, splits ri) := TRANSFORM
		  SELF.node_id := (le.node_id << 1) + IF(le.value<ri.value,0,1);
		  SELF.id := le.id;
		END;
		// The ,LOOKUP means that the result will be distributed by ID still
		ndata := JOIN(working_nodes,splits,LEFT.node_id = RIGHT.node_id AND LEFT.number=RIGHT.number,NoteNI(LEFT,RIGHT),LOOKUP);
		// The we apply those record_id/node_id pairs back to the original data / we can use local because of the ,LOOKUP above
		patched := JOIN(working_nodes,ndata,LEFT.id=RIGHT.id,TRANSFORM(Node,SELF.node_id := RIGHT.node_id, SELF.level := LEFT.level+1,SELF := LEFT),LOCAL);
    // leafs
    leafs1 := JOIN(working_nodes, leafspans, LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOOKUP);
    leafs2 := JOIN(working_nodes, pass, LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOOKUP);
    RETURN nodes(level<p_level)+leafs1+ leafs2+splits+patched;
  END;
	  d1 := DISTRIBUTE(PROJECT(ML.Utils.Fat(f,0),TRANSFORM(Node,SELF.Level := 1, SELF.node_id := 1,SELF := LEFT)),HASH(id));
		SHARED Res := LOOP(D1,Depth,Split(ROWS(LEFT),COUNTER));
		EXPORT Splits := Res(id=0); // The split points used to partition each node id
		EXPORT Partitioned := Res(id<>0); // The training data - all partitioned
		EXPORT Counts := TABLE(Partitioned(number=1),{ node_id, Cnt := COUNT(GROUP) }, node_id, FEW); // Number of training elements in each partition
		EXPORT CountMean := AVE(Counts,Cnt);
		EXPORT CountVariance := VARIANCE(Counts,Cnt);
		EXPORT Extents := TABLE(Partitioned,{ node_id, number, MinV := MIN(GROUP,Value), MaxV := MAX(GROUP,Value) }, node_id, number, FEW);
    completeTree(DATASET(Node) nodes):= FUNCTION
      leftChildren:= PROJECT(nodes, TRANSFORM(node, SELF.node_id:= LEFT.node_id << 1, SELF.level := LEFT.level+1, SELF:=[]), LOCAL);
      rightChildren:=PROJECT(nodes, TRANSFORM(node, SELF.node_id:= (LEFT.node_id << 1) + 1, SELF.level := LEFT.level+1, SELF:=[]), LOCAL);
      x:=JOIN(splits, leftChildren + rightChildren, LEFT.node_id = RIGHT.node_id, TRANSFORM(RIGHT), RIGHT ONLY);
      RETURN nodes + x;
    END;
    EXPORT FullTree := DISTRIBUTE(completeTree(Splits),HASH(node_id));  // All splits nodes must have 2 children (another split or leaf node)
    nodeBoundaries(DATASET(Node) nodes):= FUNCTION
      loopbody(DATASET(sNode) nodes):= FUNCTION
        itself:= PROJECT(nodes, TRANSFORM(sNode, SELF.Id:= LEFT.node_id *2 + IF(LEFT.HighBranch, 1, 0), SELF:=LEFT), LOCAL);
        parentNodeID := PROJECT(nodes, TRANSFORM(sNode, SELF.node_id:= LEFT.node_id DIV 2, SELF.HighBranch:= (LEFT.node_id % 2)=1, SELF:=LEFT), LOCAL);
        parentData := JOIN(splits, parentNodeID, LEFT.node_id=RIGHT.NODE_id, TRANSFORM(sNode, SELF.splitId:= RIGHT.splitId, SELF.HighBranch:= RIGHT.HighBranch, SELF:= LEFT));
        RETURN itself + parentData;
      END;
      loop0:= PROJECT(nodes,TRANSFORM(sNode, SELF.splitId:= LEFT.node_id, SELF:=LEFT), LOCAL);
      allBounds := LOOP(loop0, LEFT.id=0 AND LEFT.level>0,loopbody(ROWS(LEFT)));
      LowBounds := DEDUP(SORT(allBounds(node_id<>splitId, HighBranch=FALSE), splitId, number, -level),splitId, number);
      UpBounds  := DEDUP(SORT(allBounds(node_id<>splitId, HighBranch=TRUE) , splitId, number, -level),splitId, number);
      RETURN SORT(LowBounds + UpBounds, splitId, -level); //(node_id<>splitId)
    END;
    EXPORT Boundaries:= nodeBoundaries(FullTree);
    EXPORT LowBounds:=  Boundaries(HighBranch=TRUE);
    EXPORT UpBounds:=   Boundaries(HighBranch=FALSE);
		EXPORT NewInstances(DATASET(ML.Types.NumericField) newData, DATASET(Node) model= Splits):= MODULE
			maxLevel:= MAX(model, level);
			instSplitRec := RECORD
				ML.Types.t_node node_id; // The node-id for a given point
				ML.Types.t_level level;
				ML.Types.t_FieldNumber snumber;
				ML.Types.t_FieldReal svalue;
				ML.Types.NumericField;
			END;
			loopbody(DATASET(Node) nodes, ML.Types.t_level p_level) := FUNCTION
				allresult:= JOIN(newData, nodes, LEFT.id = RIGHT.id, TRANSFORM(Node, SELF.number:= LEFT.number, SELF.value:= LEFT.value, SELF:=RIGHT), MANY LOOKUP);
				joinall := JOIN(allresult, model, LEFT.number = RIGHT.number AND LEFT.node_id = RIGHT.node_id AND RIGHT.level= p_level, TRANSFORM(instSplitRec, SELF.node_id:= RIGHT.node_id, SELF.level:= RIGHT.level, SELF.snumber:= RIGHT.number, SELF.svalue:= RIGHT.value, SELF:= LEFT), ALL);
				leaf:= JOIN(nodes, model, LEFT.node_id = RIGHT.node_id, LOOKUP, LEFT ONLY);
				split:= PROJECT(joinall, TRANSFORM(Node, SELF.node_id := (LEFT.node_id << 1) + IF(LEFT.value < LEFT.svalue,0,1), SELF.level:= LEFT.level + 1, SELF:= LEFT));
				RETURN leaf + split;
			END;
			root:= PROJECT(newData(number=1), TRANSFORM(Node, SELF.level:= 1, SELF.node_id:=1, SELF:= LEFT));
			EXPORT Locations:= LOOP(root, maxLevel, loopbody(ROWS(LEFT),COUNTER));
		END;
	END;

// loopbody function used by SplitsGiniImpurBased split function
  EXPORT PartitionGiniImpurityBased	(DATASET(NodeInstDiscrete) nodes, ML.Types.t_level p_level, REAL Purity=1.0) := FUNCTION
    node_base := MAX(nodes,node_id); // Start allocating new node-ids from the highest previous
    // Getting only one attribute data per Node
    min_attr  := TABLE(nodes, {node_id, atr:= MIN(GROUP, number)}, node_id, MERGE);
    this_set0 := JOIN(nodes, min_attr, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.atr,TRANSFORM(LEFT),LOOKUP);
    // Computing the purities for each node, and identifying pure enough Nodes
    Purities := ML.Utils.Gini(this_set0,node_id,depend);
    PureEnough := Purities(1-Purity >= gini);
		// Removing 'pure enough' nodes from the working set and getting them through
    this_set := JOIN(nodes,PureEnough,LEFT.node_id=RIGHT.node_id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
    pass_thru := JOIN(this_set0,PureEnough,LEFT.node_id=RIGHT.node_id,TRANSFORM(LEFT),LOOKUP);
    // Computing Gini Impurity for each node-split posibility
    agg  := TABLE(this_set,{node_id,number,value,depend,Cnt := COUNT(GROUP)},node_id,number,value,depend,MERGE);
    aggc := TABLE(agg,{node_id,number,value,TCnt := SUM(GROUP,Cnt)},node_id,number,value,MERGE);
    r := RECORD
      agg;
      REAL4 Prop; // Proportion pertaining to dependent value
    END;
		prop := JOIN(agg,aggc,LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value, TRANSFORM(r, SELF.Prop := LEFT.Cnt/RIGHT.Tcnt, SELF := LEFT),HASH);
		gini_per := TABLE(prop,{node_id,number,value,tcnt := SUM(GROUP,Cnt),val := SUM(GROUP,Prop*Prop)},node_id,number,value); // 1-gini coefficient for each node for each field for each value
		gini := TABLE(gini_per,{node_id,number,gini_t := SUM(GROUP,tcnt*val)/SUM(GROUP,tcnt)},node_id,number,FEW); 		// weighted average to give the impurity based upon the field
    // Selecting best attribute to split per node and allocating instances in new nodes
    splt := DEDUP(SORT(DISTRIBUTE(gini, HASH(node_id)), node_id, -gini_t, LOCAL ), node_id, LOCAL); // best split attribute
		node_bas_info := JOIN(aggc, splt, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number, TRANSFORM(LEFT), LOOKUP); // getting back node info
    node_branches := PROJECT(node_bas_info,TRANSFORM({node_bas_info, ML.Types.t_node new_nodeid},SELF.new_nodeid := node_base + COUNTER, SELF := LEFT)); // generating branches ingo => new_nodes
		node_splits := PROJECT(node_branches,TRANSFORM(NodeInstDiscrete,SELF.value := LEFT.new_nodeid,SELF.depend := LEFT.value,SELF.level := p_level, SELF.support:=LEFT.TCnt,SELF := LEFT,SELF := [])); // Construct fake NodeInstances to pass out splitting information
    r1 := RECORD
      ML.Types.t_Recordid id;
      ML.Types.t_node nodeid;
      ML.Types.t_FieldNumber number;  
    END;
    mapp := JOIN(this_set,node_branches,LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value, TRANSFORM(r1,SELF.id:= LEFT.id, SELF.nodeid:=RIGHT.new_nodeid, SELF.number:= RIGHT.number),LOOKUP);
    // Now use the mapping to actually reset all the points, selected attribute info not going thru for next iteration
    newNodeInstances := JOIN(this_set,mapp,LEFT.id=RIGHT.id AND LEFT.number<>RIGHT.number, TRANSFORM(NodeInstDiscrete,SELF.node_id:=RIGHT.nodeid,SELF.level:=LEFT.level+1,SELF := LEFT),LOCAL);
    RETURN newNodeInstances + node_splits + pass_thru;
  END;
  
// Splitting Function Based on Gini Impurity, builds a decision tree
// Previously implemented in Decision MODULE by David Bayliss, changed to return a dataset with branch nodes and final nodes */
	EXPORT SplitsGiniImpurBased(DATASET(ML.Types.DiscreteField) ind,DATASET(ML.Types.DiscreteField) dep, ML.Types.t_level Depth=10,REAL Purity=1.0) := FUNCTION
		ind0 := DISTRIBUTE(ind, HASH32(id));
		dep0 := DISTRIBUTE(dep, HASH32(id));
    NodeInstDiscrete init(ind0 le,dep0 ri) := TRANSFORM
      SELF.node_id := 1;
      SELF.level := 1;
      SELF.depend := ri.value;	// Actually copies the dependant value to EVERY node - paying memory to avoid downstream cycles
      SELF := le;
    END;
    ind1 := JOIN(ind0, dep0, LEFT.id = RIGHT.id, init(LEFT,RIGHT), LOCAL); // Instance data distributed by HASH32(id) allows LOCAL operations	
    res := LOOP(ind1, Depth -1, LEFT.level=COUNTER, PartitionGiniImpurityBased(ROWS(LEFT), COUNTER, Purity));
    branches := PROJECT(res(id=0),TRANSFORM(SplitF, SELF.new_node_id := LEFT.value, SELF.value := LEFT.depend, SELF := LEFT)); // The split points used to partition each node i
    mode_r := RECORD
      res.node_id;
      res.level;
      res.depend;
      support := COUNT(GROUP);
    END;
    nsplits := TABLE(res(id<>0), mode_r, node_id, level, depend, FEW);
    leafs:= PROJECT(nsplits, TRANSFORM(SplitF, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN branches + leafs; 
  END;
  
  EXPORT SplitSparseGI(DATASET(ML.Types.DiscreteField) indep, DATASET(ML.Types.DiscreteField) dep, REAL Purity=1.0, defValue=0) := FUNCTION
    dIndep := DISTRIBUTE(indep, HASH(id));
    dDep   := DISTRIBUTE(  dep, HASH(id));
    anyid  := dDep[1].id;
    attnum := 5; // COUNT(dIndep(id = anyid));
    NodeInstDiscrete init(dDep ldep) := TRANSFORM
      SELF.node_id := 1;
      SELF.level   := 1;
      SELF.depend  := ldep.value;
      SELF         := ldep;
    END;
    root := PROJECT(dDep, init(LEFT), LOCAL);
    PartitionGiniImpurity	(DATASET(NodeInstDiscrete) nodes, ML.Types.t_level p_level) := FUNCTION
      node_base:= MAX(nodes, node_id);
      // Compute the purities for each node
      // Calculating Gini Impurity for each node
      aggNode   := TABLE(nodes,   {node_id, depend, cnt := COUNT(GROUP)}, node_id, depend, MERGE);
      aggNodec  := TABLE(aggNode, {node_id,     tcnt := SUM(GROUP, cnt)}, node_id,         MERGE);
      recNode   := RECORD
        aggNode;
        REAL4 prop; // Proportion pertaining to its dependant value
      END;
      propNode  := JOIN(aggNode, aggNodec, LEFT.node_id = RIGHT.node_id, TRANSFORM(recNode, SELF.prop := LEFT.cnt/RIGHT.tcnt, SELF := LEFT), HASH);
      Purities  := TABLE(propNode, {node_id, TotalCnt := SUM(GROUP,Cnt), gini := 1-SUM(GROUP, prop*prop)}, node_id); // Gini Impurity and support for each node
      // At this level these nodes are pure enough
      PureEnough := Purities(1-Purity >= gini);
      // Remove the 'pure enough' from the working set
      this_set0 := JOIN(nodes,PureEnough,LEFT.node_id=RIGHT.node_id,TRANSFORM(LEFT),LEFT ONLY,LOOKUP);
      // New working set after removing pass through node-instances
      nodes_toSplit := DISTRIBUTE(this_set0, HASH(id));
      // Populating nodes' attributes to split
      toSplit := JOIN(dIndep(value >0), nodes_toSplit, LEFT.id = RIGHT.id, TRANSFORM(NodeInstDiscrete,
              SELF.number:= LEFT.number; SELF.value:= LEFT.value; SELF:= RIGHT;), LOCAL);
      aggBrValDep  := TABLE(toSplit, {node_id, number, value, depend, Cnt := COUNT(GROUP)}, node_id, number, value, depend, MERGE);
      aggBrDep  := TABLE(aggBrValDep, {node_id, number, depend,     TCnt  := SUM(GROUP, Cnt)}, node_id, number, depend,     MERGE);
      aggBrTot  := TABLE(aggBrDep, {node_id, number,                  Tot := SUM(GROUP, TCnt)}, node_id, number,            MERGE);
      aggDefDep  := JOIN(aggBrTot, aggNode, LEFT.node_id=RIGHT.node_id,
                        TRANSFORM(RECORDOF(aggBrValDep), SELF.value:= defValue, SELF.depend:= RIGHT.depend, SELF.Cnt:= RIGHT.Cnt, SELF:=LEFT), LOOKUP, MANY);
      aggBrDef   := JOIN(aggDefDep, aggBrDep, LEFT.node_id=RIGHT.node_id AND LEFT.number = RIGHT.number AND LEFT.depend=RIGHT.depend,
                        TRANSFORM(RECORDOF(aggBrValDep), SELF.Cnt:= LEFT.Cnt - RIGHT.TCnt, SELF:=LEFT), LOOKUP, LEFT OUTER, LOOKUP); 
      aggBranch  := aggBrValDep + aggBrDef;
      aggBrCum   := TABLE(aggBranch, {node_id, number, value, tot := SUM(GROUP, Cnt)}, node_id, number, value,       MERGE);
      r := RECORD
        aggBranch;
        REAL4 Prop; // Proportion pertaining to its dependent value
      END;
      // Calculating Gini Impurity after every split
      prop      := JOIN(aggBranch, aggBrCum, LEFT.node_id = RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value,
                      TRANSFORM(r, SELF.Prop := LEFT.Cnt/RIGHT.tot, SELF := LEFT));
      gini_per  := TABLE(prop, {node_id, number, value, tcnt := SUM(GROUP,Cnt),val := 1-SUM(GROUP,Prop*Prop)}, node_id, number, value);
      gini      := TABLE(gini_per, {node_id, number, gini_t := SUM(GROUP,tcnt*val)/SUM(GROUP,tcnt)}, node_id, number);
      // Selecting the split with minimum Gini Impurity per node
      splits    := DEDUP( SORT( DISTRIBUTE( gini,HASH(node_id) ), node_id, gini_t, LOCAL ), node_id, LOCAL );
      // PureEnough nodes are transformed to Leaf Nodes and returned via pass_through
      pass_thru := JOIN(aggNode, splits, LEFT.node_id=RIGHT.node_id, TRANSFORM(NodeInstDiscrete,
                        SELF.level:= p_level, SELF.support:=LEFT.cnt, SELF.id:=0, SELF.number:=0, SELF.value:=0, SELF:=LEFT), LEFT ONLY, LOOKUP);      
      // new split nodes found
      new_spl0  := JOIN(aggBrCum(tot>0), splits, LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOOKUP);
      new_split := PROJECT(new_spl0, TRANSFORM(NodeInstDiscrete, SELF.value:= node_base + COUNTER; SELF.depend:= LEFT.value;
                                     SELF.level:= p_level; SELF.support:= LEFT.tot; SELF := LEFT; SELF := [];));
      node_inst0 := JOIN(toSplit, new_split(depend<>defValue), LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.depend,
                      TRANSFORM(NodeInstDiscrete, SELF.node_id:=RIGHT.value, SELF.level:= RIGHT.level +1, SELF.value:= LEFT.depend, SELF:= LEFT), LOOKUP);
      ni_notval := JOIN(nodes_toSplit, node_inst0, LEFT.id = RIGHT.id, TRANSFORM(LEFT),LEFT ONLY, LOCAL);
      node_inst1 := JOIN(ni_notval, new_split(depend=defValue), LEFT.node_id=RIGHT.node_id,
                      TRANSFORM(NodeInstDiscrete, SELF.node_id:=RIGHT.value, SELF.number:=RIGHT.number, SELF.level:= RIGHT.level +1, SELF:= LEFT), LOOKUP);      
      node_inst:= node_inst0 +node_inst1;
      
      RETURN pass_thru + new_split + node_inst;   // returning leaf nodes, new splits nodes and reassigned instances
    END;

    res := LOOP(root, LEFT.level=COUNTER, COUNTER < attnum, PartitionGiniImpurity(ROWS(LEFT), COUNTER));
    // Turning LOOP results into splits and leaf nodes
    SplitF toNewNode(NodeInstDiscrete NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.value, 0);
      SELF.number       := IF(NodeInst.number>0, NodeInst.number, 0);
      SELF.value        := NodeInst.depend;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT));    // node splits and leaf nodes
    mode_r := RECORD
      res.node_id;
      res.level;
      res.depend;
      support := COUNT(GROUP);
    END;
    depCnt := TABLE(res(id<>0), mode_r, node_id, level, depend, FEW);
    maxlev_leafs:= PROJECT(depCnt, TRANSFORM(SplitF, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes(support >0) + maxlev_leafs;
  END;
    
// Splitting Function Based on Information Gain Ratio,
// Returns a dataset with branch nodes and final nodes
  EXPORT SplitsInfoGainRatioBased(DATASET(ML.Types.DiscreteField) indep, DATASET(ML.Types.DiscreteField) dep) := FUNCTION
    dIndep := DISTRIBUTE(indep, HASH(id));
    dDep   := DISTRIBUTE(  dep, HASH(id));
    anyid  := dDep[1].id;
    attnum := COUNT(dIndep(id = anyid));
    NodeInstDiscrete init(dDep ldep) := TRANSFORM
      SELF.node_id := 1;
      SELF.level   := 1;
      SELF.depend  := ldep.value;
      SELF         := ldep;
		END;
    root := PROJECT(dDep, init(LEFT), LOCAL);
    // BodyFunction to split a set of nodes based on Information Gain Ratio
    PartitionInfoGainRatioBased	(DATASET(NodeInstDiscrete) nodes, ML.Types.t_level p_level) := FUNCTION
      node_base:= MAX(nodes, node_id);
      // Calculating Information Entropy of Nodes
      top_dep := TABLE(nodes, {node_id, depend, cnt:= COUNT(GROUP)}, node_id, depend, MERGE);
      top_dep_tot := TABLE(top_dep, {node_id, tot:= SUM(GROUP, cnt)}, node_id, MERGE);
      tdp := RECORD
        top_dep;
        REAL4 prop; // Proportion based only on dependent value
        REAL4 plogp:= 0;
      END;
      P_Log_P(REAL P) := IF(P=1, 0, -P*LOG(P)/LOG(2));
      top_dep_p:= JOIN(top_dep, top_dep_tot, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(tdp, SELF.prop:= LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot), SELF:=LEFT), LOOKUP);
      top_info := TABLE(top_dep_p, {node_id, info:= SUM(GROUP, plogp)}, node_id, MERGE); // Nodes Information Entropy
      PureNodes := top_info(info = 0); // Pure Nodes have Info Entropy = 0
      // Node-instances in pure nodes pass through
      pass_thru:= JOIN(top_dep, PureNodes, LEFT.node_id=RIGHT.node_id, TRANSFORM(NodeInstDiscrete,
                       SELF.node_id:= LEFT.node_id, SELF.level:= p_level, SELF.depend:=LEFT.depend, SELF.support:=LEFT.cnt,
                       SELF.id:=0, SELF.number:=0, SELF.value:=0), LOOKUP);
      // New working set after removing pass through node-instances
      nodes_toSplit := DISTRIBUTE(JOIN(nodes, PureNodes, LEFT.node_id=RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP), HASH(id));
      // Populating nodes' attributes to split
      toSplit := JOIN(dIndep, nodes_toSplit, LEFT.id = RIGHT.id, TRANSFORM(NodeInstDiscrete,
              SELF.number:= LEFT.number; SELF.value:= LEFT.value; SELF:= RIGHT;), LOCAL);
      // Calculating Information Gain of possible splits
      child := TABLE(toSplit, {node_id, number, value, depend, cnt := COUNT(GROUP)}, node_id, number, value, depend,MERGE);
      child_tot:= TABLE(child, {node_id, number, value, tot := SUM(GROUP,cnt)}, node_id, number, value, MERGE);
      csp := RECORD
        child_tot;
        REAL4 prop;
        REAL4 plogp;
      END;
      // Calculating Intrinsic Information Entropy of each attribute(split) per node
      csplit_p:= JOIN(child_tot, top_dep_tot, LEFT.node_id = RIGHT.node_id,
                TRANSFORM(csp, SELF.prop:= LEFT.tot/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.tot/RIGHT.tot), SELF:=LEFT));
      csplit:= TABLE(csplit_p, {node_id, number, split_info:=SUM(GROUP, plogp)},node_id, number, MERGE); // Intrinsic Info
      chp := RECORD
        child;
        REAL4 prop; // Proportion pertaining to this dependant value
        REAL4 plogp:= 0;
      END;
      // Information Entropy of new branches per split
      cprop := JOIN(child, child_tot, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value,
              TRANSFORM(chp, SELF.prop := LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot), SELF:=LEFT));
      cplogp := TABLE(cprop, {node_id, number, value, cont:= SUM(GROUP,cnt), inf0:= SUM(GROUP, plogp)}, node_id, number, value);
      // Information Entropy of possible splits per node
      cinfo := TABLE(cplogp, {node_id, number, info:=SUM(GROUP, cont*inf0)/SUM(GROUP, cont)}, node_id, number);
      grec := RECORD
        ML.Types.t_node node_id;
        ML.Types.t_Discrete number;
        REAL4 gain;
      END;
      // Information Gain of possible splits per node
      gain := JOIN(cinfo, top_info, LEFT.node_id=RIGHT.node_id,
                TRANSFORM(grec, SELF.node_id:= LEFT.node_id, SELF.number:=LEFT.number, SELF.gain:= RIGHT.info - LEFT.info), LOOKUP);
      grrec := RECORD
        ML.Types.t_node node_id;
        ML.Types.t_Discrete number;
        REAL4 gain_ratio;
      END;
      // Information Gain Ratio of possible splits per node
      gainRatio := JOIN(gain, csplit, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number,
              TRANSFORM(grrec, SELF.node_id:= LEFT.node_id, SELF.number:=LEFT.number, SELF.gain_ratio:= LEFT.gain/RIGHT.split_info));
      // Selecting the split with max Info Gain Ratio per node
      split:= DEDUP(SORT(DISTRIBUTE(gainRatio, HASH(node_id)), node_id, -gain_ratio, LOCAL), node_id, LOCAL);
      // new split nodes found
      new_spl0  := JOIN(child_tot, split, LEFT.node_id = RIGHT.node_id AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOOKUP);
      new_split := PROJECT(new_spl0, TRANSFORM(NodeInstDiscrete, SELF.value:= node_base + COUNTER; SELF.depend:= LEFT.value;
                                     SELF.level:= p_level; SELF.support:= LEFT.tot; SELF := LEFT; SELF := [];));
      node_inst := JOIN(toSplit, new_split, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.depend,
                      TRANSFORM(NodeInstDiscrete, SELF.node_id:=RIGHT.value, SELF.level:= RIGHT.level +1, SELF.value:= LEFT.depend, SELF:= LEFT ));
      RETURN pass_thru + new_split + node_inst;   // returning leaf nodes, new splits nodes and reassigned instances
    END;
    res := LOOP(root, LEFT.level=COUNTER, COUNTER < attnum, PartitionInfoGainRatioBased(ROWS(LEFT), COUNTER));
    // Turning LOOP results into splits and leaf nodes
    SplitF toNewNode(NodeInstDiscrete NodeInst) := TRANSFORM
      SELF.new_node_id  := IF(NodeInst.number>0, NodeInst.value, 0);
      SELF.number       := IF(NodeInst.number>0, NodeInst.number, 0);
      SELF.value        := NodeInst.depend;
      SELF:= NodeInst;
    END;
    new_nodes:= PROJECT(res(id=0), toNewNode(LEFT));    // node splits and leaf nodes
    mode_r := RECORD
      res.node_id;
      res.level;
      res.depend;
      support := COUNT(GROUP);
    END;
    depCnt := TABLE(res(id<>0), mode_r, node_id, level, depend, FEW);
    maxlev_leafs:= PROJECT(depCnt, TRANSFORM(SplitF, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN new_nodes + maxlev_leafs;
  END;
  
  // Still used in C45PruneTree, it doesn't use weights to handle new values not it the model
  // It will be deleted after C45PruneTree upgrade
	EXPORT SplitInstances(DATASET(Splitf) mod, DATASET(ML.Types.DiscreteField) Indep) := FUNCTION
			splits:= mod(new_node_id <> 0);	// separate split or branches
			leafs := mod(new_node_id = 0);	// from final nodes
			join0 := JOIN(Indep, splits, LEFT.number = RIGHT.number AND LEFT.value = RIGHT.value, LOOKUP, MANY);
			sort0 := SORT(join0, id, level, number, node_id, new_node_id);
			dedup0:= DEDUP(sort0, LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT);
			dedup1:= DEDUP(dedup0, LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT);
			RETURN dedup1;
	END;
  EXPORT SplitInstancesD(DATASET(Splitf) mod, DATASET(ML.Types.DiscreteField) Indep) := FUNCTION
    inst_node:= RECORD(ML.Types.DiscreteField)
      ML.Types.t_node       node_id;
      ML.Types.t_level      level;
      ML.Types.t_FieldReal  weight;
    END;
    depth:=MAX(mod, level);
    root:= mod(node_id =1);
    root_att:= MAX(root, number);
    ind0:= DISTRIBUTE(Indep, HASH32(id));
    rootInstances:= PROJECT(ind0(number = root_att), TRANSFORM(inst_node, SELF.node_id:=1, SELF.level:=1, SELF.weight:=1.0, SELF:= LEFT), LOCAL);
    wwNode := RECORD
      ML.Types.t_node node_id; // The node-id for a given point
      ML.Types.t_level level; // The level for a given point
      ML.Types.t_FieldNumber number; // The column used to split
      ML.Types.t_Discrete value; // The value for the column in question
      ML.Types.t_node new_node_id:=0; // The new node that value goes to
      ML.Types.t_FieldReal weight;
    END;
    accmod := TABLE(mod, {node_id, tot:= SUM(GROUP, support)},node_id);
    wNodes := JOIN(mod, accmod, LEFT.node_id=RIGHT.node_id, TRANSFORM(wwNode, SELF.weight:=LEFT.support/RIGHT.tot, SELF:=LEFT), LOOKUP);
    loop_body(DATASET(inst_node) inst_nodes, ML.Types.t_level p_level) := FUNCTION
      dinst_nod:= DISTRIBUTE(inst_nodes, HASH32(id)); 
      instances:= JOIN(dinst_nod, ind0, LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number, TRANSFORM(inst_node, SELF.value:=RIGHT.value, SELF:= LEFT), LOCAL); 
      nodesN   := wNodes(level=p_level);
      matchN   := JOIN(instances, nodesN, LEFT.node_id=RIGHT.node_id AND LEFT.level = RIGHT.level AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value, LOOKUP);
      no_match := JOIN(instances, matchN, LEFT.node_id=RIGHT.node_id AND LEFT.id=RIGHT.id, LEFT ONLY, LOCAL);
      match_val:= PROJECT(matchN, TRANSFORM(inst_node, SELF.node_id:=LEFT.new_node_id, SELF.level:=LEFT.level+1, SELF:=LEFT), LOCAL);
      miss_val := JOIN(no_match, nodesN, LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.node_id:=RIGHT.new_node_id, SELF.level:=LEFT.level+1, SELF.weight:= LEFT.weight*RIGHT.weight, SELF:=LEFT), LOOKUP, MANY);
      all_val  := miss_val + match_val;
      nodesNext:= wNodes(level=p_level+1);
      go_brnch := JOIN(all_val, nodesNext(number<>0), LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.number:= RIGHT.number, SELF.value:=RIGHT.value, SELF:= LEFT), LOOKUP);
      go_leaf  := JOIN(all_val, nodesNext(number=0), LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.number:= RIGHT.number, SELF.value:=RIGHT.value, SELF.weight:= LEFT.weight*RIGHT.weight, SELF:= LEFT), LOOKUP, MANY);
      RETURN go_brnch + go_leaf;
    END;
    RETURN LOOP(rootInstances, depth, LEFT.number>0, loop_body(ROWS(LEFT), COUNTER));
  END;
  
  EXPORT ClassProbDistribD(DATASET(ML.Types.DiscreteField) Indep,DATASET(ML.Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitF, nodes, model_Map);
    results:= SplitInstancesD(nodes, Indep);
    acc_result:= RECORD
      results.id;
      results.value;
      REAL8 conf:= SUM(GROUP, results.weight);
    END;
    acc_results:= TABLE(results, acc_result, id, value);
    RETURN PROJECT(acc_results, TRANSFORM(ML.Types.l_result, SELF.number:= 1, SELF:= LEFT, SELF:=[]));
  END;
  
  EXPORT SplitSparseInstD(DATASET(Splitf) mod, DATASET(ML.Types.DiscreteField) Indep, defValue=0) := FUNCTION
    inst_node:= RECORD(ML.Types.DiscreteField)
      ML.Types.t_node       node_id;
      ML.Types.t_level      level;
      ML.Types.t_FieldReal  weight;
    END;
    depth    := MAX(mod, level);
    root     := mod(node_id =1);
    root_att := MAX(root, number);
    ind0     := DISTRIBUTE(Indep, HASH32(id));
    root_ins := PROJECT(ind0(number=0), TRANSFORM(inst_node, SELF.number:= root_att, SELF.value:= defValue, SELF.node_id:=1, SELF.level:=1, SELF.weight:=1.0, SELF:= LEFT), LOCAL);
    wwNode := RECORD
      ML.Types.t_node node_id; // The node-id for a given point
      ML.Types.t_level level; // The level for a given point
      ML.Types.t_FieldNumber number; // The column used to split
      ML.Types.t_Discrete value; // The value for the column in question
      ML.Types.t_node new_node_id:=0; // The new node that value goes to
      ML.Types.t_FieldReal weight;
    END;
    accmod := TABLE(mod, {node_id, tot:= SUM(GROUP, support)},node_id);
    wNodes := JOIN(mod, accmod, LEFT.node_id=RIGHT.node_id, TRANSFORM(wwNode, SELF.weight:=LEFT.support/RIGHT.tot, SELF:=LEFT), LOOKUP);
    loop_body(DATASET(inst_node) inst_nodes, ML.Types.t_level p_level) := FUNCTION
      dinst_nod:= DISTRIBUTE(inst_nodes, HASH32(id));
      instances:= JOIN(dinst_nod, ind0, LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number, TRANSFORM(inst_node, SELF.value:= IF(RIGHT.number=0, defValue, RIGHT.value), SELF:= LEFT), LEFT OUTER, LOCAL);
      nodesN   := wNodes(level=p_level);
      matchN   := JOIN(instances, nodesN, LEFT.node_id=RIGHT.node_id AND LEFT.level = RIGHT.level AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value, LOOKUP);
      no_match := JOIN(instances, matchN, LEFT.node_id=RIGHT.node_id AND LEFT.id=RIGHT.id, LEFT ONLY, LOCAL);
      match_val:= PROJECT(matchN, TRANSFORM(inst_node, SELF.node_id:=LEFT.new_node_id, SELF.level:=LEFT.level+1, SELF:=LEFT), LOCAL);
      miss_val := JOIN(no_match, nodesN, LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.node_id:=RIGHT.new_node_id, SELF.level:=LEFT.level+1, SELF.weight:= LEFT.weight*RIGHT.weight, SELF:=LEFT), LOOKUP, MANY);
      all_val  := miss_val + match_val;
      nodesNext:= wNodes(level=p_level+1);
      go_brnch := JOIN(all_val, nodesNext(number<>0), LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.number:= RIGHT.number, SELF.value:=RIGHT.value, SELF:= LEFT), LOOKUP);
      go_leaf  := JOIN(all_val, nodesNext(number=0), LEFT.node_id=RIGHT.node_id, TRANSFORM(inst_node, SELF.number:= RIGHT.number, SELF.value:=RIGHT.value, SELF.weight:= LEFT.weight*RIGHT.weight, SELF:= LEFT), LOOKUP, MANY);
      RETURN go_brnch + go_leaf;
    END;
    RETURN LOOP(root_ins, depth, LEFT.number>0, loop_body(ROWS(LEFT), COUNTER));
  END;
  
  EXPORT ClassProbDistSparseD(DATASET(ML.Types.DiscreteField) Indep,DATASET(ML.Types.NumericField) mod, defValue=0) := FUNCTION
    ML.FromField(mod, SplitF, nodes, model_Map);
    results:= SplitSparseInstD(nodes, Indep, defValue);
    acc_result:= RECORD
      results.id;
      results.value;
      REAL8 conf:= SUM(GROUP, results.weight);
    END;
    acc_results:= TABLE(results, acc_result, id, value);
    RETURN PROJECT(acc_results, TRANSFORM(ML.Types.l_result, SELF.number:= 1, SELF:= LEFT, SELF:=[]));
  END;
//Function that prune a Decision Tree based on Estimated Error (C4.5 Quinlan)
//Inputs:
//		- nodes dataset from a learning process
//		- Independent and Depenedent datasets, should not use the same used in the learning process
//		- z score corresponding to confidence factor, default z = 0.67449
//			default confidence factor = 0.25 -> the positive z for 2 * 0.25% confidence interval = between -0.67449 and 0.67449 
	EXPORT C45PruneTree(DATASET(Splitf) nodes, DATASET(ML.Types.DiscreteField) Indep, DATASET(ML.Types.DiscreteField) Dep, REAL z=0.67449):= FUNCTION
		splitData	:= SplitInstances(nodes, Indep); 	// splits the instances throughout the tree (looking for a leaf node, same as classify)
		branches	:= nodes(new_node_id <> 0);				// identify branch nodes
		leafs 		:= nodes(new_node_id = 0);				// identify leaf nodes
		max_level	:= MAX(nodes, level);
		// Calculate the N x estimated error of a leaf node
		node_error NxErrEst(node_error l):= TRANSFORM
			UNSIGNED4 N:= l.cnt;
			REAL4 f:= l.e/N;
			SELF.NxErr_est:= N*(f + z*z/(2*N) + z*SQRT(f/N - (f*f)/N + (z*z)/(4*N*N)) )/(1 + z*z/N);
			SELF:=l;
		END;
		// Store split_instance/final_node info
		final_node_instance final_nodes(RECORDOF(splitData) l, RECORDOF(leafs) r ):= TRANSFORM
			SELF.instance_id 		:= l.id;
			SELF.root_node 			:= l.node_id;
			SELF.root_level			:= l.level;
			SELF.final_node			:= r.node_id;
			SELF.final_level		:= r.level;
			SELF.final_class		:= r.value;
		END;
		// Store instance_class value and evaluate match with final_class value
		final_node_instance actual_class(final_node_instance l, ML.Types.DiscreteField r):= TRANSFORM
			SELF.instance_class	:= r.value;
			SELF.match					:= l.final_class = r.value;
			SELF:= l;
		END;
		// Return all children nodes of a branch
		ExplodeSubTree(DATASET(SplitF) branch):= FUNCTION
			local_level:= MAX(branch, level);
			RETURN branch + JOIN(nodes, branch(level = local_level), LEFT.node_id=RIGHT.new_node_id);
		END;
		// Return parent's node_id of a node
		FindParent(ML.Types.t_RecordID child_id):= FUNCTION
			parent_node:= branches(new_node_id = child_id);
			RETURN if(exists(parent_node), MAX(parent_node,node_id), 0);
		END;
		// Populating instance-nodes with predicted and actual classes, and calcualting leaf nodes Error Estimated
		class_as:= JOIN(splitData, leafs, LEFT.new_node_id = RIGHT.node_id, final_nodes(LEFT, RIGHT), LOOKUP); // classified as
		real_class:= JOIN(class_as, Dep, LEFT.instance_id = RIGHT.id, actual_class(LEFT, RIGHT), LOOKUP);	// real classes and matches
		rc_err:= TABLE(real_class, {root_node, root_level, final_node, final_level, final_class, e:= COUNT(GROUP,real_class.match=FALSE), cnt:= COUNT(GROUP)}, root_node, root_level, final_node, final_level, final_class);
		leaf_error:=PROJECT(rc_err, node_error);
		leaf_NxErrEst:= PROJECT(leaf_error, NxErrEst(LEFT)); // Calculate N x error estimated on tree's leaf nodes

		// loop body of instance split accumulated
		loopbody1(DATASET(final_node_instance) nodes_inst, INTEGER1 p_level) := FUNCTION
			this_set:= nodes_inst(root_level = p_level);
			final_node_instance trx(SplitF l, final_node_instance r):= TRANSFORM
				SELF.root_node	:= l.node_id;
				SELF.root_level	:= l.level;
				SELF.final_node := r.root_node;
				SELF.final_level:= r.root_level;
				SELF:= r;
			END;
			join1:= JOIN(nodes, this_set, left.new_node_id = right.root_node, trx(LEFT, RIGHT));
			RETURN nodes_inst + join1;
		END;
		// Generating possible replacing nodes for each branch -> repo nodes
		acc_split:= LOOP(real_class, max_level, loopbody1(ROWS(LEFT), max_level - COUNTER)); //  instance splits accumulated
		g_acc_split:= TABLE(acc_split, {root_node, root_level, final_class, cnt:= COUNT(GROUP)}, root_node, root_level, final_class);
		gtot_acc_split:= TABLE(g_acc_split,{root_node, root_level, tot:=SUM(GROUP, cnt)}, root_node, root_level);
		g_join:= JOIN(g_acc_split, gtot_acc_split, LEFT.root_node = RIGHT.root_node AND LEFT.root_level = RIGHT.root_level, TRANSFORM(node_error,
				SELF.root_node:= FindParent(LEFT.root_node), SELF.root_level:= LEFT.root_level -1,
				SELF.final_node:= LEFT.root_node, SELF.final_level:= LEFT.root_level,
				SELF.final_class:=LEFT.final_class, SELF.e:= RIGHT.tot - LEFT.cnt, SELF.cnt:=RIGHT.tot));
		g_sort:= SORT(g_join, final_node, e);		// sorting based on error to chose the final class with less errors
		repo_nodes:= DEDUP(g_sort, final_node); // repo nodes
		repo_NxErrEst:= PROJECT(repo_nodes,NxErrEst(LEFT)); // Calculate N x error estimated on repo nodes
		
		// loop body of branchs and repo nodes error estimated comparisson
		loopbody2(DATASET(node_error) all_nodes, INTEGER1 n_level) := FUNCTION
			level_nodes:= all_nodes(root_level = n_level);	// get only level nodes
			// calculating error estimated of branch (all leaf nodes with same root_node)
			g_level_nodes:= TABLE(level_nodes,{root_node, root_level, err:=SUM(GROUP, e), tot:= SUM(GROUP, cnt), totErr_est:=SUM(GROUP, nxerr_est)}, root_node, root_level);
			// transforming to update upper level error estimated value
			lnodes_NxErrEst:= PROJECT(g_level_nodes, TRANSFORM(node_error,
					SELF.root_node:= FindParent(LEFT.root_node), SELF.root_level:=LEFT.root_level -1,
					SELF.final_node:= LEFT.root_node, SELF.final_level:=LEFT.root_level,
					SELF.e:= LEFT.err, SELF.cnt:= LEFT.tot, SELF.nxerr_est:= LEFT.toterr_est, SELF.final_class:= -1));
			level_repo:= repo_NxErrEst(final_level= n_level);	// get the repo nodes for the level
			to_chose:=SORT(lnodes_NxErrEst + level_repo, final_node, nxerr_est);
			for_update:=DEDUP(to_chose, final_node);		// will update dataset with chosen nodes
			for_delete:= for_update(final_class >= 0);	// extract leafs nodes from chosen nodes
			to_delete:=PROJECT(for_delete, TRANSFORM(SplitF, SELF.node_id:=LEFT.root_node, SELF.level:= LEFT.root_level, SELF.new_node_id:=LEFT.final_node, SELF:=[]));
			subtree_del := LOOP(to_delete, max_level, ExplodeSubTree(ROWS(LEFT)));	// will erase whole subtrees of deleted nodes
			pass_thru_nodes:= JOIN(all_nodes, subtree_del, LEFT.root_node= RIGHT.node_id AND LEFT.final_node=RIGHT.new_node_id, TRANSFORM(LEFT), LEFT ONLY);
			RETURN pass_thru_nodes + for_update;
		END;
		// Comparing branches to repo nodes,
		// if repo node has smaller error estimated than branch, delete branch and replace
		// else generate upper level branch node with NxErrEst updated
		new_nodes:= LOOP(leaf_NxErrEst, max_level -1, loopbody2(ROWS(LEFT), max_level - COUNTER));
		Splitf to_leaf(node_error ne):= TRANSFORM
			SELF.node_id	:= ne.final_node;
			SELF.level		:= ne.final_level;
			SELF.number		:= 0;
			SELF.value		:= ne.final_class;
			SELF.new_node_id:= 0;
		END;
		// transforming results into Spliitf records to return
		new_branches:= JOIN(nodes, new_nodes, LEFT.node_id = RIGHT.root_node AND LEFT.new_node_id = RIGHT.final_node, TRANSFORM(LEFT), LOOKUP);
		new_leafs:= PROJECT(new_nodes(final_class > -1), to_leaf(LEFT));
		RETURN new_branches + new_leafs;
	END;

//Function that generates a DT and then prunes it using C45PruneTree
//The function receives Independent and Dependet datasets,
//	fold them into numFolds folds, minimum 3
//	use numFolds - 1 folds to generate the unpruned DT,
//	use 1 fold to prune to DT.
//Return the pruned DT.
	EXPORT SplitsIGR_Pruned(DATASET(ML.Types.DiscreteField) Indep, DATASET(ML.Types.DiscreteField) Dep, INTEGER1 numFolds = 3, REAL z = 0.67449 ):=FUNCTION
		folds:= ML.Sampling.DiscreteDepNFolds(Dep, MAX(3, numFolds));
		trainIndep:= JOIN(Indep, folds(number = 1), LEFT.id = RIGHT.id, LEFT ONLY);
		trainDep	:= JOIN(Dep, folds(number = 1), LEFT.id = RIGHT.id, LEFT ONLY);
		testIndep	:= JOIN(Indep, folds(number > 1), LEFT.id = RIGHT.id, LEFT ONLY);
		testDep		:= JOIN(Dep, folds(number > 1), LEFT.id = RIGHT.id, LEFT ONLY);
		raw_tree:= SplitsInfoGainRatioBased(trainIndep, trainDep);

		RETURN C45PruneTree(raw_tree, testIndep, testDep);
	END;
  EXPORT ToDiscreteTree(DATASET(SplitD) nodes) := FUNCTION
    ML.AppendID(nodes, id, model);
    ML.ToField(model, out_model, id, model_fields);
    RETURN out_model;
  END;
  EXPORT ModelD(DATASET(ML.Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitD,o, model_Map);
    RETURN o;
  END;
  EXPORT ClassifyD(DATASET(ML.Types.DiscreteField) Indep,DATASET(ML.Types.NumericField) mod) := FUNCTION
    // getting class probabilities for each instance
    results:= ClassProbDistribD(Indep, mod);
    dClass := DISTRIBUTE(results, HASH32(id)); 
    // selecting the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN PROJECT(finalClass, TRANSFORM(ML.Types.l_result, SELF:= LEFT, SELF:=[]), LOCAL);
  END;

  EXPORT ClassifySparseD(DATASET(ML.Types.DiscreteField) Indep,DATASET(ML.Types.NumericField) mod, defValue=0) := FUNCTION
    // getting class probabilities for each instance
    results:= ClassProbDistSparseD(Indep, mod, defValue);
    dClass := DISTRIBUTE(results, HASH32(id)); 
    // selecting the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN PROJECT(finalClass, TRANSFORM(ML.Types.l_result, SELF:= LEFT, SELF:=[]), LOCAL);
  END;

//  Methods to handle Continuous Data with Decision Trees

  // Function that splits the tree (continuos independent values) based on Info Gain Ratio crtiteria
  EXPORT BinaryPartitionC	(DATASET(cNode) nodes, ML.Types.t_level p_level, ML.Types.t_Count minNumObj=2) := FUNCTION
    node_base:= MAX(nodes, node_id);
    nodes_level:= nodes(level = p_level);
    this_set := DISTRIBUTE(nodes_level, HASH(node_id, number));
    root_sorted := SORT(this_set, node_id, number, value, depend, LOCAL);
    attrib1:= root_sorted(number=1);
    node_dep := TABLE(attrib1, {node_id, depend, high_fork, cnt:= COUNT(GROUP)}, node_id, depend, high_fork, FEW);
    // Calculating total count and minSplit(minimun number of instances per Split) parameter values  for each node
    node_dep_tot := TABLE(node_dep, {node_id, tot:= SUM(GROUP, cnt), numclass:= COUNT(GROUP), REAL minSplit:= MIN(25,MAX(minNumObj, 0.1*(SUM(GROUP, cnt)/COUNT(GROUP))))}, node_id, FEW);
    // Propagating total and minSplit to every row per class
    node_dep_all := JOIN(node_dep, node_dep_tot, LEFT.node_id = RIGHT.node_id);
    // NoSplit Nodes: nodes with not enough instances to be split or with only one class value
    root_noSplit := node_dep_tot(tot < (2*minSplit) OR numclass =1);
    root_NoSplit_node := DEDUP(SORT(JOIN(root_noSplit, node_dep, LEFT.node_id = RIGHT.node_id,
                                        TRANSFORM(RIGHT), MANY LOOKUP), node_id, -cnt), node_id);
    // Transforming NoSplit Nodes into LEAF Nodes to return
    pass_pure_NoSplit:= PROJECT(root_NoSplit_node, TRANSFORM(cNode, SELF.id:= 0, SELF.number:= 0, SELF.value:= 0, SELF.level:= p_level, SELF:=LEFT));
    // Calculating Information Entropy of Nodes before split, continue
    tdp := RECORD
      node_dep;
      REAL4 prop; // Proportion based only on dependent value
      REAL4 plogp:= 0;
    END;
    P_Log_P(REAL P) := IF(P=1, 0, -P*LOG(P)/LOG(2));
    node_dep_p:= JOIN(node_dep, node_dep_tot, LEFT.node_id = RIGHT.node_id,
        TRANSFORM(tdp, SELF.prop:= LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot), SELF:=LEFT), LOOKUP);
    // Information Entropy of nodes before split
    node_entropy := TABLE(node_dep_p, {node_id, info:= SUM(GROUP, plogp)}, node_id);
    // Compact Impure Node's data to unique node-attribute values
    root_impure_all:= JOIN(root_sorted, root_noSplit, LEFT.node_id = RIGHT.node_id, TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    root_acc_dep := TABLE(root_impure_all, {node_id, number, value, depend, depcnt:=COUNT(GROUP)}, node_id, number, value, depend, MERGE);
    root_acc_distrib := DISTRIBUTE(root_acc_dep, HASH(node_id, number));  // Distribution in order to speed up counting
    root_acc:= TABLE(root_acc_distrib, {node_id, number, value, cut_cnt:=SUM(GROUP, depcnt)}, node_id, number, value, LOCAL);
    // Set of all posible split points (total counts and split Info initialized with 0)
    rec_cut:= RECORD
       root_acc;
       INTEGER tot_Low:=0;  // number of ocurrences <= treshold
       INTEGER tot_High:=0; // number of ocurrences > treshold
       INTEGER tot:=0;      // number of ocurrences
       REAL minSplit;       // minimum number of occurrences needed to perform a Split
       REAL split_Info:=0;  // Intrinsic Information Entropy of splits
    END;
    cuts:= JOIN(root_acc, node_dep_tot, LEFT.node_id = RIGHT.node_id,
        TRANSFORM(rec_cut, SELF.tot:= RIGHT.tot, SELF.minSplit:= RIGHT.minSplit, SELF:=LEFT), LOOKUP);
    sort_cuts:= SORT(cuts, node_id, number, value, LOCAL);
    rec_cut rol(sort_cuts le, sort_cuts ri) := TRANSFORM
      t_low:=   ri.cut_Cnt + IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      t_high:=  ri.tot - ri.cut_Cnt - IF(le.node_id=ri.node_id AND le.number=ri.number , le.tot_Low, 0);
      p_low:=   t_low/ri.tot;
      p_high:=  t_high/ri.tot;
      SELF.tot_Low:= t_low;
      SELF.tot_High:= t_high;
      SELF.split_Info:= P_Log_P(p_low) + P_Log_P(p_high);
      SELF := ri;
    END;
    // Accumulated Counting: t_low # ocurrences <= treshold , t_high # ocurrences > treshold
    x := ITERATE(sort_cuts, rol(LEFT,RIGHT), LOCAL);
    // Filtering cuts with not enough occurrences needed to perform a Split
    cuts_ok:= x((tot_Low >= minSplit) AND (tot_High >= minSplit));
    nodes_ok:= TABLE(cuts_ok, {node_id}, node_id, MERGE);
    cuts_noSplit:= TABLE(x((tot_Low < minSplit ) OR (tot_High < minSplit)), {node_id}, node_id, MERGE);
    // Nodes with none acceptable splits become LEAFS
    node_noSplit:= JOIN(cuts_noSplit, nodes_ok, LEFT.node_id=RIGHT.node_id, LEFT ONLY, LOOKUP);
    noSplit_dep := DEDUP(SORT(JOIN(node_noSplit, node_dep, LEFT.node_id = RIGHT.node_id,
                                  TRANSFORM(RIGHT), MANY LOOKUP), node_id, -cnt), node_id);
    pass_thru_noSplit:= PROJECT(noSplit_dep, TRANSFORM(cNode, SELF.level:= p_level, SELF:=LEFT, SELF:=[]));
    // Calculating MDL Correction: Minimum Description Length principle (Rissanen, 1983),
    bag_count := TABLE(cuts_ok, {node_id, number, bagCnt:= COUNT(GROUP)}, node_id, number, LOCAL);
    MDLcorrection:= JOIN(bag_count, node_dep_tot, LEFT.node_id=RIGHT.node_id,
        TRANSFORM({bag_Count, REAL IGpenalty}, SELF.IGpenalty:= (LOG(LEFT.bagCnt)/LOG(2))/RIGHT.tot, SELF:=LEFT), LOOKUP);

    // Set of all possible bags (based on cuts tresholds and dependent counts, total counts initialized with 0)
    // The bags are used to calculate Information Entropy of all possible splits per node
    rec_dep:= RECORD
       root_acc_dep;
       INTEGER tot_Low:=0;    // total number of ocurrences of Dependent with attrib-value <= treshold the Bag
       INTEGER tot_High:=0;   // total number of ocurrences of Dependent with attrib-value > treshold the Bag
       INTEGER tot_Dep:=0;    // total number of ocurrences of Dep value at the Bag
       INTEGER tot_Node:=0;   // total number of ocurrences at the Node
       REAL minSplit;
    END;
    rec_dep pop_dep(root_acc_dep le, node_dep_all ri):= TRANSFORM
      SELF.depend:=   ri.depend;
      SELF.depcnt:=   IF(le.depend= ri.depend, le.depcnt, 0);
      SELF.tot_Dep:=  ri.cnt;
      SELF.tot_Node:= ri.tot;
      SELF:=          le;
      SELF:=          ri;
    END;
    deps:= JOIN(root_acc_dep, node_dep_all, LEFT.node_id = RIGHT.node_id, pop_dep(LEFT, RIGHT), MANY LOOKUP);
    sort_deps:= SORT(deps, node_id, number, depend, value, LOCAL);
    rec_dep rold(sort_deps le, sort_deps ri) := TRANSFORM
      SELF.tot_Low:= ri.depCnt + IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend , le.tot_Low, 0);
      SELF.tot_High:= ri.tot_dep - ri.depCnt - IF(le.node_id=ri.node_id AND le.number=ri.number AND le.depend=ri.depend, le.tot_Low, 0);
      SELF := ri;
    END;
    // Accumulated Counting per Dependent value per Cut threshold
    bag_grouped := ITERATE(sort_deps, rold(LEFT,RIGHT), LOCAL);
    bag_dep := RECORD
      bag_grouped;
      REAL4 prop_low;   // Proportion pertaining to dependant value
      REAL4 plogp_low:= 0;
      REAL4 prop_high;  // Proportion pertaining to dependant value
      REAL4 plogp_high:= 0;
    END;
    // Information Entropy of new branches per split
    bag_gprop:= JOIN(bag_grouped, cuts_ok, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value = RIGHT.value,
        TRANSFORM(bag_dep, SELF.prop_low := LEFT.tot_low/RIGHT.tot_low, SELF.plogp_low:= P_LOG_P(LEFT.tot_low/RIGHT.tot_low),
                  SELF.prop_high := LEFT.tot_high/RIGHT.tot_high, SELF.plogp_high:= P_LOG_P(LEFT.tot_high/RIGHT.tot_high), SELF:=LEFT), HASH);
    bag_gplogp := TABLE(bag_gprop, {node_id, number, value, cont_low:= SUM(GROUP, tot_low), inf_low:= SUM(GROUP, plogp_low),
                                    cont_high:= SUM(GROUP, tot_high), inf_high:= SUM(GROUP, plogp_high), tot_node}, node_id, number, value, LOCAL);
    // Information Entropy of possible splits per node
    bag_info := RECORD
      ML.Types.t_node node_id;
      ML.Types.t_FieldNumber number;
      ML.Types.t_FieldReal value;
      ML.Types.t_FieldReal info;
    END;
    bags_Entropy:=PROJECT(bag_gplogp, TRANSFORM(bag_info, SELF.info:= (LEFT.cont_low/LEFT.tot_node)*LEFT.inf_low
                                    + (LEFT.cont_high/LEFT.tot_node)*LEFT.inf_high, SELF:=LEFT), LOCAL);
    // Calculating Information Gain of possible splits
    bag_gr := RECORD
      bags_Entropy;
      REAL4 gain:= 0;
      REAL4 g_ratio:= 0;
    END;
    bags_gain:= JOIN(bags_Entropy, node_entropy, LEFT.node_id=RIGHT.node_id,
        TRANSFORM(bag_gr, SELF.gain:= RIGHT.info - LEFT.info , SELF:= LEFT), LOOKUP);
    // Selecting the split with max Info Gain per bag
    bagSplit:= DEDUP(SORT(bags_gain, node_id, number, -gain, LOCAL), node_id, number, LOCAL);
    bagsplit_MDL:= JOIN(bagSplit, MDLcorrection, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number,
        TRANSFORM(bag_gr, SELF.gain:=LEFT.gain-RIGHT.IGpenalty, SELF:=LEFT), LOCAL);
    bagsplit_MDL_acc:=TABLE(bagsplit_MDL,{node_id, posCnt:=COUNT(GROUP, gain>0), totCnt:=COUNT(GROUP)},node_id);
    // Nodes with none acceptable MDL become LEAFS
    node_noMDL:=PROJECT(bagsplit_MDL_acc(posCnt=0), TRANSFORM({ML.Types.t_node node_id}, SELF:= LEFT));
    noMDL_dep := DEDUP(SORT(JOIN(node_noMDL, node_dep, LEFT.node_id = RIGHT.node_id, TRANSFORM(RIGHT)), node_id, -cnt), node_id);
    pass_thru_noMDL:= PROJECT(noMDL_dep, TRANSFORM(cNode, SELF.level:= p_level, SELF:=LEFT, SELF:=[]));
    // Selecting the split with max Gain Ratio per node from acceptable MDL splits
    bagSplit_gRatio := JOIN(cuts_ok, bagsplit_MDL(gain > 0), LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND LEFT.value=RIGHT.value,
        TRANSFORM(bag_gr, SELF.info:= LEFT.split_info, SELF.g_ratio:= RIGHT.gain/LEFT.split_info, SELF:= RIGHT), LOCAL);
    node_splits:= DEDUP(SORT(bagSplit_gRatio, node_id, -g_ratio), node_id);
    // Start allocating new node-ids from the highest previous
    new_nodes_low:= PROJECT(node_splits, TRANSFORM(cNode, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER -1, SELF.level:= p_level, SELF.high_fork:=FALSE, SELF := LEFT));
    new_nodes_high:= PROJECT(node_splits, TRANSFORM(cNode, SELF.id:= 0, SELF.value:= LEFT.value, SELF.depend := node_base+ 2*COUNTER, SELF.level:= p_level, SELF.high_fork:=TRUE, SELF := LEFT));
    new_nodes:= new_nodes_low + new_nodes_high;
    // Assignig instances that didn't reach a leaf node to (new) node-ids (by joining to the sampled data)
    leafsNodes:= pass_pure_NoSplit + pass_thru_noSplit;
    noleaf:= JOIN(root_impure_all, leafsNodes, LEFT.node_id = RIGHT.node_id, LEFT ONLY, LOOKUP);
    r1 := RECORD
      ML.Types.t_Recordid id;
      ML.Types.t_node nodeid;
      BOOLEAN high_fork:=FALSE;
    END;
    mapp := JOIN(noleaf, new_nodes, LEFT.node_id=RIGHT.node_id AND LEFT.number=RIGHT.number AND (LEFT.value>RIGHT.value)= RIGHT.high_fork,
                TRANSFORM(r1, SELF.id := LEFT.id, SELF.nodeid:=RIGHT.depend, SELF.high_fork:=RIGHT.high_fork ),LOOKUP);
    // Now use the mapping to actually reset all the points
    J := JOIN(noleaf, mapp,LEFT.id=RIGHT.id,TRANSFORM(cNode, SELF.node_id:=RIGHT.nodeid, SELF.level:=LEFT.level+1, SELF.high_fork:=RIGHT.high_fork, SELF := LEFT));
    RETURN nodes(level < p_level) + leafsNodes + new_nodes + J + pass_thru_noMDL;
  END;

  // Function that learn from Numeric data and builds a Binary Decision Tree based on Info Gain Ratio
  //    minNumObj   minimum number of instances in a leaf node, used in splitting process
  //    maxLevel    stop learning criteria, either tree's level reachs maxLevel depth or no more split can be done.
  EXPORT SplitBinaryCBased(DATASET(ML.Types.NumericField) Indep, DATASET(ML.Types.DiscreteField) Dep, ML.Types.t_Count minNumObj=2, ML.Types.t_level maxLevel=32) := FUNCTION
    depth   := MIN(1023, maxLevel); // Max number of iterations when building trees (max 1023 levels)
    // ind0 := ML.Utils.Fat(Indep);    // Ensure no sparsity in independents
    ind0 := Indep;
    cNode init(ind0 le, dep ri) := TRANSFORM
      SELF.node_id := 1;
      SELF.level := 1;
      SELF.depend := ri.value;
      SELF := le;
    END;
    // All instances start at root node (node_id = 1)
    root :=JOIN(ind0, Dep, LEFT.id = RIGHT.id, init(LEFT,RIGHT)); // If we were prepared to force DEP into memory then ,LOOKUP would go quicker
    // LOOP keep going until split is not possible or the tree reachs maximum level: (~sp or ~mx <=> sp AND mx)
    res := LOOP(root, MAX(ROWS(LEFT), level)>= COUNTER AND COUNTER < depth , BinaryPartitionC(ROWS(LEFT), COUNTER, minNumObj));
    splits := PROJECT(res(id=0, number>0),TRANSFORM(SplitC, SELF.new_node_id := LEFT.depend, SELF.value := LEFT.value, SELF.high_fork:=(INTEGER1)LEFT.high_fork, SELF := LEFT));    // node splits
    leafs1 := PROJECT(res(id=0, number=0),TRANSFORM(SplitC, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF.high_fork:=(INTEGER1)LEFT.high_fork, SELF:= LEFT)); // leafs nodes
		mode_r := RECORD
			res.node_id;
			res.level;
			res.depend;
      cnt := COUNT(GROUP);
		END;
		// Taking care instances (id>0) that reached maximum level and did not turn into a leaf yet
    depCnt      := TABLE(res(id>0, number=1),mode_r, node_id, level, depend, FEW);
    // Assigning class value based on majority voting
    depCntSort  := SORT(depCnt, node_id, -cnt);    // if there is more than one dependent value for nodeid
    depCntDedup := DEDUP(depCntSort, node_id);     // then select the value of the class with more counts
    leafs2:= PROJECT(depCntDedup, TRANSFORM(SplitC, SELF.number:=0, SELF.value:= LEFT.depend, SELF.new_node_id:=0, SELF:= LEFT));
    RETURN splits + leafs1+ leafs2;
  END;
  // Function that locates instances into the deepest branch node (split) based on their attribute values
  EXPORT SplitBinInstances(DATASET(SplitC) mod, DATASET(ML.Types.NumericField) Indep) := FUNCTION
    splits:= mod(new_node_id <> 0);	// Get split nodes (branches)
    ind   := DISTRIBUTE(Indep, HASH(id));
    join0 := JOIN(ind, splits, LEFT.number = RIGHT.number AND RIGHT.high_fork = IF(LEFT.value > RIGHT.value, 1, 0), LOOKUP, MANY);
    sort0 := SORT(join0, id, level, number, node_id, new_node_id, LOCAL);
    dedup0:= DEDUP(sort0, LEFT.id = RIGHT.id AND LEFT.new_node_id != RIGHT.node_id, KEEP 1, LEFT, LOCAL);
    dedup1:= DEDUP(dedup0, LEFT.id = RIGHT.id AND LEFT.new_node_id = RIGHT.node_id, KEEP 1, RIGHT, LOCAL);
    RETURN dedup1;
  END;
  EXPORT ToNumericTree(DATASET(SplitC) nodes) := FUNCTION
    ML.AppendID(nodes, id, model);
    ML.ToField(model, out_model, id, modelC_fields);
    RETURN out_model;
  END;
  EXPORT ModelC(DATASET(ML.Types.NumericField) mod) := FUNCTION
    ML.FromField(mod, SplitC,o, modelC_Map);
    RETURN o;
  END;
  EXPORT ClassifyC(DATASET(ML.Types.NumericField) Indep,DATASET(ML.Types.NumericField) mod) := FUNCTION
    // Transform NumericFiled "mod" to Trees.SplitC "nodes" model format using field map modelC_Map
    ML.FromField(mod, SplitC, nodes, modelC_Map);
    leafs := nodes(new_node_id = 0);                    // Get leaf nodes from model
    // Locate instances into deepest split node based upon independent values
    splitData:= SplitBinInstances(nodes, Indep);
    // Locate instances into final leaf node and get Dependent value
    ML.Types.l_result final_class(RECORDOF(splitData) l, RECORDOF(leafs) r ):= TRANSFORM
      SELF.id 		:= l.id;
      SELF.number	:= 1;
      SELF.value	:= r.value;
      SELF.conf		:= 0;		// added to fit in l_result, not used so far
    END;
    RETURN JOIN(splitData, leafs, LEFT.new_node_id = RIGHT.node_id, final_class(LEFT, RIGHT), LOOKUP);
  END;
END;