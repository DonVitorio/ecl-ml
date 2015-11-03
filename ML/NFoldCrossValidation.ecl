IMPORT * FROM ML;
IMPORT * FROM ML.Types;
EXPORT NFoldCrossValidation(IndepDS, DepDS, LearnerName, NumFolds) := FUNCTIONMACRO
  learner:= LearnerName;
  #DECLARE(fields);
  #DECLARE(ilearn) #SET(ilearn,'')
  #DECLARE(icpd)   #SET(icpd,'')
  #DECLARE(iclass) #SET(iclass,'')
  #EXPORTXML(fields, RECORDOF(IndepDS));
  #FOR(fields)
    #FOR(Field)
      #IF(%'{@label}'% = 'value') 
        #IF(%'{@type}'% = 'integer') 
          #SET(ilearn, 'RETURN learner.LearnD(indData, depData); \n')
          #SET(icpd,   'RETURN learner.ClassProbDistribD(indData, mod); \n')
          #SET(iclass, 'RETURN learner.ClassifyD(indData, mod); \n')
        #ELSE
          #SET(ilearn, 'RETURN learner.LearnC(indData, depData); \n')
          #SET(icpd,   'RETURN learner.ClassProbDistribC(indData, mod); \n')
          #SET(iclass, 'RETURN learner.ClassifyC(indData, mod); \n')
        #END
      #END
    #END
  #END
  
  AUC_Rec:= RECORD(ML.Types.NumericField)
    DATASET(ML.Classify.AUCcurvePoint) curvePoints;
  END;
  idFoldRec := RECORD
    Types.t_FieldNumber fold;
    Types.t_RecordID id;
  END;
  dsRecordRnd := RECORD(ML.Types.DiscreteField)
    Types.t_FieldNumber rnd:= 0;
  END; 
  dsRecordRnd AddRandom(ML.Types.DiscreteField l) :=TRANSFORM
    SELF.rnd := RANDOM();
    SELF := l;
  END;
  Learn(DATASET(RECORDOF(IndepDS)) indData, DATASET(Types.DiscreteField) depData) := FUNCTION
    #EXPAND(%'ilearn'%)
  END;
  ClassProbDistrib(DATASET(RECORDOF(IndepDS)) indData, DATASET(Types.NumericField) mod) := FUNCTION
    #EXPAND(%'icpd'%)
  END;
  Classify(DATASET(RECORDOF(IndepDS)) indData, DATASET(Types.NumericField) mod) := FUNCTION
    #EXPAND(%'iclass'%)    
  END;
  FoldNDS(DATASET(RECORDOF(IndepDS)) indData, DATASET(Types.DiscreteField) depData, DATASET(idFoldRec) ds_folds, Types.t_Discrete num_fold, Types.t_RecordID baseId = 0) := MODULE
    EXPORT trainIndep := JOIN(indData, ds_folds(fold <> num_fold), LEFT.id = RIGHT.id, TRANSFORM(RECORDOF(IndepDS), SELF.id:= LEFT.id + baseId, SELF:=LEFT), LOCAL);
    EXPORT trainDep   := JOIN(depData, ds_folds(fold <> num_fold), LEFT.id = RIGHT.id, TRANSFORM(Types.DiscreteField, SELF.id:= LEFT.id + baseId, SELF.number:= num_fold, SELF:=LEFT), LOCAL);
    EXPORT testIndep  := JOIN(indData, ds_folds(fold = num_fold), LEFT.id = RIGHT.id, TRANSFORM(RECORDOF(IndepDS), SELF.id:= LEFT.id + baseId, SELF:=LEFT), LOCAL);
    EXPORT testDep    := JOIN(depData, ds_folds(fold = num_fold), LEFT.id = RIGHT.id, TRANSFORM(Types.DiscreteField, SELF.id:= LEFT.id + baseId, SELF.number:= num_fold, SELF:=LEFT), LOCAL);
  END;
  toFoldResult(DATASET(ML.Types.l_result) iCPD, DATASET(ML.Types.l_result) iClass, ML.Types.t_Discrete num_fold) := MODULE
    EXPORT CPD := PROJECT(iCPD, TRANSFORM(ML.Types.l_result, SELF.number:= num_fold; SELF:= LEFT), LOCAL);
    EXPORT Class := PROJECT(iClass, TRANSFORM(ML.Types.l_result, SELF.number:= num_fold; SELF:= LEFT), LOCAL);
  END; 
  dRnd := PROJECT(DepDS, AddRandom(LEFT), LOCAL);
  dRndSorted := SORT(dRnd,value,rnd);
  ds_parts := DISTRIBUTE(PROJECT(dRndSorted, TRANSFORM(idFoldRec, SELF.fold := COUNTER%NumFolds + 1, SELF:= LEFT)), id);
  dIndep  := DISTRIBUTE(IndepDS, id);
  dDep    := DISTRIBUTE(DepDS, id);
//  classes := TABLE(dDep,{number, value}, number, value);
  #DECLARE (SetString)    #SET (SetString, '');
  #DECLARE (SetLearner)   #SET (SetLearner, REGEXREPLACE('[^a-zA-Z0-9_.,()]',#TEXT(LearnerName),''));
  #DECLARE (Ndx)
  #SET (Ndx, 1);
  #LOOP
    #IF (%Ndx% > NumFolds)  
       #BREAK         // break out of the loop
    #ELSE             //otherwise
      #APPEND(SetString,'fold'     + %'Ndx'% + ':= FoldNDS(dIndep, dDep, ds_parts, ' + %'Ndx'% + '); \n');
      #APPEND(SetString,'indepN'   + %'Ndx'% + ':= fold' + %'Ndx'% + '.trainIndep; \n');
      #APPEND(SetString,'depN'     + %'Ndx'% + ':= fold' + %'Ndx'% + '.trainDep; \n');
      #APPEND(SetString,'t_indepN' + %'Ndx'% + ':= fold' + %'Ndx'% + '.testIndep; \n');
      #APPEND(SetString,'t_depN'   + %'Ndx'% + ':= fold' + %'Ndx'% + '.testDep; \n');
      #APPEND(SetString,'modN'     + %'Ndx'% + ':= Learn(indepN' + %'Ndx'% + ', depN' + %'Ndx'% + '); \n');
      #APPEND(SetString,'cpdN'     + %'Ndx'% + ':= ClassProbDistrib(t_indepN' + %'Ndx'% + ', modN' + %'Ndx'% + '); \n');
      #APPEND(SetString,'classN'   + %'Ndx'% + ':= Classify(t_indepN' + %'Ndx'% + ', modN' + %'Ndx'% + '); \n');
      #APPEND(SetString,'tfResN'   + %'Ndx'% + ':= toFoldResult(cpdN' + %'Ndx'% + ', classN' + %'Ndx'% + ',' + %'Ndx'% + '); \n');
      #SET (Ndx, %Ndx% + 1)  //and increment the value of Ndx
    #END
  #END
  #EXPAND(%'SetString'%);
  #SET (Ndx, 1);
  #DECLARE (aggCPD)   #SET (aggCPD,   'allCPD:= ');
  #DECLARE (aggClass) #SET (aggClass, 'allClass:= ');
  #DECLARE (aggtDep)  #SET (aggtDep,  'alltDep:= ');
  #LOOP
    #IF (%Ndx% < NumFolds)  
      #APPEND(aggCPD,   'tfResN' + %'Ndx'% + '.CPD +');
      #APPEND(aggClass, 'tfResN' + %'Ndx'% + '.Class +');
      #APPEND(aggtDep,  't_depN' + %'Ndx'% + ' +');
      #SET (Ndx, %Ndx% + 1)  //and increment the value of Ndx
    #ELSE             //otherwise
      #APPEND(aggCPD,   'tfResN' + %'Ndx'% + '.CPD; \n');
      #APPEND(aggClass, 'tfResN' + %'Ndx'% + '.Class; \n');
      #APPEND(aggtDep, 't_depN'  + %'Ndx'% + '; \n');
      #BREAK         // break out of the loop
    #END
  #END
  #EXPAND(%'aggCPD'%);
  #EXPAND(%'aggClass'%);
  #EXPAND(%'aggtDep'%);

  CVResults(DATASET(ML.Types.DiscreteField) it_depN, DATASET(ML.Types.l_result) iCPDN, DATASET(ML.Types.l_result) iclassN) := MODULE
    tclass:= TABLE(it_depN, {value}, value);
    fclass:= PROJECT(tclass, TRANSFORM(ML.Types.NumericField, SELF.id:= COUNTER, SELF.number:= LEFT.value, SELF.value:= 0));
    AUC_Rec GetAUC(ML.Types.NumericField L) := TRANSFORM
        SELF.curvePoints:= ML.Classify.AUC_ROC(iCPDN, L.number, it_depN);
        SELF := L;
    END;
    AUC_class:= PROJECT(fclass, GetAUC(LEFT));
    ML.Classify.AUCcurvePoint GetCurvePoints(ML.Classify.AUCcurvePoint R) := TRANSFORM
      SELF := R;
    END;
    EXPORT AUC_curvePoints  := NORMALIZE(AUC_class, LEFT.curvePoints, GetCurvePoints(RIGHT));
    EXPORT AUC_scores       := TABLE(AUC_curvePoints, {posClass, classifier, AUC_score:= MAX(GROUP, auc)}, posClass, classifier);
    SHARED TestModule       := ML.Classify.Compare(it_depN, iclassN);
    EXPORT CrossAssignments := TestModule.CrossAssignments;
    EXPORT RecallByClass    := TestModule.RecallByClass;
    EXPORT PrecisionByClass := TestModule.PrecisionByClass;
    EXPORT FP_Rate_ByClass  := TestModule.FP_Rate_ByClass;
    EXPORT Accuracy         := TestModule.Accuracy;
    EXPORT CPDN             := iCPDN;
    EXPORT classN           := iclassN;    
  END;
  RETURN CVResults(alltDep, allCPD, allClass);
ENDMACRO;