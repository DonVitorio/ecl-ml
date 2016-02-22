IMPORT * FROM ML;
IMPORT * FROM ML.Types;
EXPORT NaiveBayes := MODULE
/* Naive Bayes Classification 
   This method can support producing classification results for multiple classifiers at once
   Note the presumption that the result (a weight for each value of each field) can fit in memory at once
*/
  SHARED Base := ML.Utils.Base;
  SHARED SampleCorrection := 1;
  SHARED LogScale(REAL P) := -LOG(P)/LOG(2);

  SHARED ClassifierFeatureClass := RECORD
    Types.t_Discrete    class_number;   // Classifier ID
    Types.t_discrete    c;              // Dependent "value" value - Class value
    Types.t_FieldNumber number;         // A reference to a feature (or field) in the independants
  END;
  SHARED BayesResult := RECORD
    Types.t_RecordId    id := 0;        // A record-id - allows a model to have an ordered sequence of results
    ClassifierFeatureClass;
    Types.t_Count       Support;        // Number of cases
  END;
  SHARED BayesResultD := RECORD (BayesResult)
    Types.t_discrete  f := 0;           // Independant value - Attribute value
    Types.t_FieldReal PC;                // Either P(F|C) or P(C) if number = 0. Stored in -Log2(P) - so small is good :)
  END;
  SHARED BayesResultC := RECORD (BayesResult)
    Types.t_FieldReal  mu:= 0;          // Independent attribute mean (mu)
    Types.t_FieldReal  var:= 0;         // Independent attribute sample standard deviation (sigma squared)
  END;
  SHARED TripleD := RECORD
    ClassifierFeatureClass;
    Types.t_Discrete f;
  END;
  SHARED TripleD form(Types.DiscreteField le, Types.DiscreteField ri) := TRANSFORM
    SELF.class_number := ri.number;
    SELF.c := ri.value;
    SELF.number := le.number;
    SELF.f := le.value;
  END;
  SHARED AggTripleD := RECORD
    TripleD;
    Types.t_Count support := 0;
  END;
  SHARED AggTripleD FatValsCount(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep):= FUNCTION
    Vals := JOIN(Indep,Dep,LEFT.id=RIGHT.id,form(LEFT,RIGHT));
    // This is the raw table - how many of each value 'f' for each field 'number' appear for each value 'c' of each classifier 'class_number'
    Cnts0 := TABLE(Vals,{c,f,number,class_number,support:=COUNT(GROUP)}, class_number,number,c,f, MERGE);
    RETURN PROJECT(Cnts0, AggTripleD);
  END;
  SHARED clCnt_Rec := RECORD
    Types.DiscreteField.number;
    Types.DiscreteField.value;
    Types.t_Discrete Support;
  END;
  SHARED AggTripleD SparseValsCount(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep, DATASET(clCnt_Rec) CTotals, Types.t_discrete defValue = 0):= FUNCTION
    Vals := JOIN(Indep,Dep,LEFT.id=RIGHT.id,form(LEFT,RIGHT));
    // This is the raw table - how many of each value 'f' for each field 'number' appear for each value 'c' of each classifier 'class_number'
    Cnts00 := TABLE(Vals,{c,f,number,class_number,support:=COUNT(GROUP)}, class_number,number,c,f, MERGE);
    Cnts0  := TABLE(Cnts00, {c,number,class_number, tsupport:=SUM(GROUP, support)}, class_number,number,c, MERGE);
    NonEmptyAtt:= TABLE(Cnts0, {number,class_number},number,class_number);
    DefCnts := JOIN(NonEmptyAtt, CTotals, LEFT.class_number=RIGHT.number, TRANSFORM(AggTripleD, SELF.f:=defValue, SELF.support:= RIGHT.support, SELF.c:= RIGHT.value, SELF:= LEFT));
    SpareCnts:= JOIN(DefCnts, Cnts0, LEFT.class_number=RIGHT.class_number AND LEFT.number=RIGHT.number AND LEFT.c=RIGHT.c, TRANSFORM(AggTripleD, SELF.support:= LEFT.support - RIGHT.tsupport , SELF:= LEFT), LEFT OUTER);
    RETURN PROJECT(Cnts00, AggTripleD) +  SpareCnts(support>0);
  END;
/*
  LearnD is used in both simple NaiveBayes and SparseNaiveBayes classifiers.
*/
  EXPORT LearnD(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep, BOOLEAN IgnoreMissing = FALSE, BOOLEAN SparseData = FALSE, Types.t_discrete defValue = 0) := FUNCTION
    dd := Indep(number>0); // Just in case it comes with the COUNT record, it is not needed to Learn.
    cl := Dep;
    // Compute P(C)
    CTots := TABLE(cl,{value,number,Support := COUNT(GROUP)},value,number,FEW);
    CLTots := TABLE(CTots,{number,TSupport := SUM(GROUP,Support), GC := COUNT(GROUP)},number,FEW);
    P_C_Rec := RECORD
      Types.t_Discrete  class_number; // Used when multiple classifiers being produced at once
      Types.t_Discrete  c;            // The value within the class
      Types.t_Discrete  f;            // The number of features within the class
      Types.t_FieldReal support;      // Used to store total number of C
      REAL8 w;                        // P(C)
    END;
    // Apply Laplace Estimator to P(C) in order to be consistent with attributes' probability
    P_C_Rec pct(CTots le,CLTots ri) := TRANSFORM
      SELF.class_number := ri.number;
      SELF.c := le.value;
      SELF.f := 0; // to be claculated later on
      SELF.support := le.Support + SampleCorrection;
      SELF.w := (le.Support + SampleCorrection) / (ri.TSupport + ri.GC*SampleCorrection);
    END;
    PC_0 := JOIN(CTots,CLTots,LEFT.number=RIGHT.number,pct(LEFT,RIGHT),FEW);
    // Computing Attributes' probability, use different functions for SparseDATA and FatDATA
    Cnts:= IF(SparseData, SparseValsCount(dd,cl,PROJECT(CTots, clCnt_Rec), defValue), FatValsCount(dd,cl));
    AttribValue_Rec := RECORD
      Cnts.class_number;  // Used when multiple classifiers being produced at once
      Cnts.number;        // A reference to a feature (or field) in the independants
      Cnts.f;             // Independant value
      Types.t_Count support := 0;
    END;
    // Generating feature domain per feature (class_number only used when multiple classifiers being produced at once)
    AttValues := TABLE(Cnts, AttribValue_Rec, class_number, number, f, FEW);
    AttCnts   := TABLE(AttValues, {class_number, number, ocurrence:= COUNT(GROUP)},class_number, number, FEW); // Summarize 
    AttValIgnoreMissing := JOIN(AttValues, AttCnts(ocurrence=1), // Filtering features with only one value, used only if IgnoreMissing = TRUE
                                LEFT.class_number = RIGHT.class_number AND LEFT.number = RIGHT.number,TRANSFORM(LEFT), LEFT ONLY, LOOKUP);
    AttrValue_per_Class_Rec := RECORD
      AttribValue_Rec;
      Types.t_Discrete c;
    END;
    // Generating class x feature domain, initial support = 0
    AttrValue_per_Class_Rec form_cl_attr(AttValues le, CTots ri):= TRANSFORM
      SELF.c:= ri.value;
      SELF:= le;
    END;
    // IgnoreMissing = TRUE will ignore features with only one value (all instance having the same value for this feature)
    ATots:= JOIN(IF(IgnoreMissing, AttValIgnoreMissing, AttValues), CTots, LEFT.class_number = RIGHT.number, form_cl_attr(LEFT, RIGHT), MANY LOOKUP, FEW);
    // Counting feature value ocurrence per class x feature, remains 0 if combination not present in dataset
    ATots form_ACnts(ATots le, Cnts ri) := TRANSFORM
      SELF.support := ri.support;
      SELF         := le;
    END;
    ACnts := JOIN(ATots, Cnts, LEFT.c = RIGHT.c AND LEFT.f = RIGHT.f AND LEFT.number = RIGHT.number AND LEFT.class_number = RIGHT.class_number, 
                          form_ACnts(LEFT,RIGHT), LEFT OUTER, LOOKUP);
    // Summarizing and getting value 'GC' to apply in Laplace Estimator
    TotalFs0 := TABLE(ACnts,{c,number,class_number,Types.t_Count Support := SUM(GROUP,Support),GC := COUNT(GROUP)},c,number,class_number,FEW);
    TotalFs  := TABLE(TotalFs0,{c,class_number,ML.Types.t_Count Support := SUM(GROUP,Support),Types.t_Count GC := SUM(GROUP,GC)},c,class_number,FEW);
    // Merge and Laplace Estimator
    F_Given_C_Rec := RECORD
      AttrValue_per_Class_Rec;
      REAL8 w;
    END;
    F_Given_C_Rec mp(ACnts le,TotalFs ri) := TRANSFORM
      SELF.support := le.Support+SampleCorrection;
      SELF.w := (le.Support+SampleCorrection) / (ri.Support+ri.GC*SampleCorrection);
      SELF := le;
    END;
    // Calculating final Feature-Class probability
    FC := JOIN(ACnts,TotalFs,LEFT.C = RIGHT.C AND LEFT.class_number=RIGHT.class_number,mp(LEFT,RIGHT),LOOKUP);
    PC_0 form_TotalFs(PC_0 le, TotalFs ri) := TRANSFORM
      SELF.f  := ri.Support+ri.GC*SampleCorrection;
      SELF    := le;
    END;
    // Calculating final Class probability
    PC    := JOIN(PC_0, TotalFs, LEFT.C = RIGHT.C AND LEFT.class_number=RIGHT.class_number,form_TotalFs(LEFT,RIGHT),LOOKUP);
    // Tranformation to BayesResultD and Probabilities to LogScale(Probabilities)
    Pret  := PROJECT(FC  ,TRANSFORM(BayesResultD, SELF.PC:=LEFT.w, SELF := LEFT)) + PROJECT(PC,TRANSFORM(BayesResultD, SELF.PC:=LEFT.w, SELF.number:= 0,SELF:=LEFT));
    Pret1 := PROJECT(Pret,TRANSFORM(BayesResultD, SELF.PC := LogScale(LEFT.PC),SELF.id := Base+COUNTER,SELF := LEFT));
    ToField(Pret1,o);
    RETURN o;
  END;
  // Transform NumericFiled "mod" to discrete Naive Bayes format model "BayesResultD"
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod,BayesResultD,o);
    RETURN o;
  END;
  Inter := RECORD
    Types.t_RecordId Id;
    Types.t_discrete class_number;
    Types.t_discrete c;
    REAL8  w;
  END;
  Inter note(Types.DiscreteField le, BayesResultD ri) := TRANSFORM
    SELF.id := le.id;
    SELF.class_number := ri.class_number;
    SELF.c  := ri.c;
    SELF.w  := ri.PC;
  END;
  InterCounted := RECORD
    Types.t_RecordId Id;
    Types.t_discrete class_number;
    Types.t_discrete c;
    REAL8 P := 0;
    Types.t_FieldNumber Missing := 0;
  END;    
  NoteMissingFat(DATASET(Types.DiscreteField) Indep, DATASET(Types.NumericField) mod, BOOLEAN IgnoreMissing = FALSE):= FUNCTION
    dd := DISTRIBUTE(Indep, HASH32(id));
    mo := Model(mod);
    // Aggregating class probabilities and Counting for Laplace Smoothing
    J := JOIN(dd, mo, LEFT.number=RIGHT.number AND LEFT.value=RIGHT.f, note(LEFT,RIGHT), MANY LOOKUP);
    TSum := TABLE(J,{c,class_number,id,P := SUM(GROUP,w),Types.t_FieldNumber Missing := COUNT(GROUP)},c,class_number,id,LOCAL);
    // Checking for instance attributes values not in the model (Missing values)
    FTots := TABLE(dd,{id,cnt := COUNT(GROUP)},id,LOCAL);
    InterCounted NoteMissing(TSum le,FTots ri) := TRANSFORM
      SELF.Missing := ri.cnt - le.Missing;
      SELF := le;
    END;
    MissingNoted := JOIN(Tsum,FTots,LEFT.id=RIGHT.id,NoteMissing(LEFT,RIGHT),LOOKUP);
    // Adding Model Class Probalilities and applying Laplace Smoothing based on IgnoreMissing flag + Missing counts
    InterCounted NoteC(MissingNoted le,mo ri) := TRANSFORM
      SELF.P := le.P + ri.PC + IF(IgnoreMissing, 0, le.Missing*LogScale(SampleCorrection/ri.f));
      SELF := le;
    END;
    RETURN JOIN(MissingNoted,mo(number=0),LEFT.c=RIGHT.c,NoteC(LEFT,RIGHT),LOOKUP);
  END;
  NoteMissingSparse(DATASET(Types.DiscreteField) Indep, DATASET(Types.NumericField) mod, BOOLEAN IgnoreMissing = FALSE, Types.t_discrete defValue = 0):= FUNCTION
    ddAll     := DISTRIBUTE(Indep, HASH32(id)); // Distributed to calculate LOCALly as much as possible
    ddDefOnly := ddAll(number=0 AND value=0);   // Instances having value = default value (defValue) for all their independent variables
    dd        := ddAll(number>0);               // Instances' independent variables with value <> default value (defValue)
    mo        := Model(mod);                    // NaiveBayes Model
    modPC     := mo(number=0);                  // Model Class Probalilities
    modAtt    := mo(number>0);                  // Model Attribute-Class Probalilities
    mo_attval := modAtt(f <> defValue);         // Attribute-Class probabilities for attributes with value <> default value
    mo_defval := modAtt(f  = defValue);         // Attribute-Class probabilities for attributes with value == default value
    // Aggregating class probabilities and Counting for Laplace Smoothing
    defSum := TABLE(mo_defval, {c, class_number, P:= SUM(GROUP, PC), Missing := COUNT(GROUP)}, c, class_number, MERGE); // Only defVal Class Probalilities from Attributes
    J0 := JOIN(ddDefOnly, defSum, TRUE, TRANSFORM(InterCounted, SELF.id:=LEFT.id, SELF.missing:=0, SELF:= RIGHT), ALL); 
    J1 := JOIN(dd, mo_attval, LEFT.number=RIGHT.number AND LEFT.value=RIGHT.f, note(LEFT,RIGHT), MANY LOOKUP);
    J2 := JOIN(dd, mo_defval, LEFT.number=RIGHT.number, note(LEFT,RIGHT), MANY LOOKUP);
    TSum1  := TABLE(J1,{c,class_number,id,P := SUM(GROUP,w),Types.t_FieldNumber Missing := COUNT(GROUP)},c,class_number,id,LOCAL); // Instances' Class Prob from Attribs with value <> default value
    TSum2  := TABLE(J2,{c,class_number,id,P := SUM(GROUP,w),Types.t_FieldNumber Missing := COUNT(GROUP)},c,class_number,id,LOCAL); // Instances' Class Prob from Attribs with value == default value
    // Subtracting and adding aggregated probabilities in order to not consider twice probabilities form defVal values
    defDiff:= JOIN(TSum2,  defSum, LEFT.class_number = RIGHT.class_number AND LEFT.c = RIGHT.c,
                                   TRANSFORM(InterCounted, SELF.P:= RIGHT.P - LEFT.P, SELF.Missing:= RIGHT.Missing - LEFT.Missing, SELF:= LEFT), LOOKUP);
    TSum   := JOIN(Tsum1, defDiff, LEFT.class_number = RIGHT.class_number AND LEFT.c = RIGHT.c AND LEFT.id = RIGHT.id, 
                                   TRANSFORM(InterCounted, SELF.P:= RIGHT.P + LEFT.P, SELF.Missing:= RIGHT.Missing + LEFT.Missing, SELF:= LEFT), LOCAL);
    FTots0 := TABLE(dd,{id,c := COUNT(GROUP)},id,LOCAL);
    // Checking for instance attributes values not in the model (Missing values)
    Ftot_Rec:= RECORD
      Types.t_RecordId Id;
      Types.t_Count   cnt;
    END;
    FTots  := JOIN(FTots0, defDiff, LEFT.id = RIGHT.id, TRANSFORM(Ftot_Rec, SELF.cnt:=LEFT.c + RIGHT.Missing, SELF:=LEFT), LOCAL);
    InterCounted NoteMissing(TSum le,FTots ri) := TRANSFORM
      SELF.Missing := ri.cnt - le.Missing;
      SELF := le;
    END;
    MissingNoted := JOIN(Tsum,FTots,LEFT.id=RIGHT.id,NoteMissing(LEFT,RIGHT),LOOKUP);
    // Adding Model Class Probalilities and applying Laplace Smoothing based on IgnoreMissing flag + Missing counts
    InterCounted NoteC(MissingNoted le,modPC ri) := TRANSFORM
      SELF.P := le.P + ri.PC + IF(IgnoreMissing, 0, le.Missing*LogScale(SampleCorrection/ri.f));
      SELF := le;
    END;
    RETURN JOIN(MissingNoted + J0, modPC,LEFT.c=RIGHT.c, NoteC(LEFT,RIGHT),LOOKUP);  
  END;
  // This function will take a pre-existing NaiveBayes model (mo) and score new instances
  // The output will have a row for every class in the original training set for each new instance 
  EXPORT ClassProbDistribD(DATASET(Types.DiscreteField) Indep, DATASET(Types.NumericField) mod, BOOLEAN IgnoreMissing = FALSE, BOOLEAN SparseData = FALSE, Types.t_discrete defValue = 0) := FUNCTION
    // Gathering Instances' class probabilities and Laplace Smoothing based on IgnoreMissing flag and Missing counts
    CNoted0:= IF(SparseData, NoteMissingSparse(Indep, mod, IgnoreMissing, defValue), NoteMissingFat(Indep, mod, IgnoreMissing));
    // dealing with floating precision
    minP    := TABLE(CNoted0, {class_number, id, pmin:= MIN(GROUP, p)}, class_number, id, LOCAL); // find minimum p per id
    CNoted  := JOIN(CNoted0, minP, LEFT.class_number = RIGHT.class_number AND LEFT.id = RIGHT.id, 
                      TRANSFORM(InterCounted, SELF.p:= LEFT.p - RIGHT.pmin, SELF:=LEFT), LOCAL);  // rebasing p before normalizing
    l_result toResult(CNoted le) := TRANSFORM
      SELF.id := le.id;               // Instance ID
      SELF.number := le.class_number; // Classifier ID
      SELF.value := le.c;             // Class value
      SELF.conf := POWER(2.0, -le.p); // Convert likehood to decimal value
    END;
    // Normalizing Likehood to deliver Class Probability per instance
    InstResults := PROJECT(CNoted, toResult(LEFT), LOCAL);
    gInst := TABLE(InstResults, {number, id, tot:=SUM(GROUP,conf)}, number, id, LOCAL);
    clDist:= JOIN(InstResults, gInst,LEFT.number=RIGHT.number AND LEFT.id=RIGHT.id, TRANSFORM(Types.l_result, SELF.conf:=LEFT.conf/RIGHT.tot, SELF:=LEFT), LOCAL);
    RETURN clDist;
  END;
  // Classification function for discrete independent values and model
  EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod, BOOLEAN IgnoreMissing = FALSE, BOOLEAN SparseData = FALSE, Types.t_discrete defValue = 0) := FUNCTION
    // get class probabilities for each instance
    dClass:= ClassProbDistribD(Indep, mod, IgnoreMissing, SparseData, defValue);
    // select the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN finalClass;
  END;
  /*From Wikipedia
  " ...When dealing with continuous data, a typical assumption is that the continuous values associated with each class are distributed according to a Gaussian distribution.
  For example, suppose the training data contain a continuous attribute, x. We first segment the data by the class, and then compute the mean and variance of x in each class.
  Let mu_c be the mean of the values in x associated with class c, and let sigma^2_c be the variance of the values in x associated with class c.
  Then, the probability density of some value given a class, P(x=v|c), can be computed by plugging v into the equation for a Normal distribution parameterized by mu_c and sigma^2_c..."
  */ 
  EXPORT LearnC(DATASET(NumericField) Indep, DATASET(DiscreteField) Dep) := FUNCTION
    TripleC := RECORD
      ClassifierFeatureClass;
      Types.t_FieldReal value;
    END;
    TripleC form(Indep le, Dep ri) := TRANSFORM
      SELF.class_number := ri.number;
      SELF.c            := ri.value;
      SELF.number       := le.number;
      SELF.value        := le.value;
    END;
    Vals := JOIN(Indep, Dep, LEFT.id=RIGHT.id, form(LEFT,RIGHT));
    // Compute P(C)
    ClassCnts := TABLE(Dep, {number, value, support := COUNT(GROUP)}, number, value, FEW);
    ClassTots := TABLE(ClassCnts,{number, TSupport := SUM(GROUP,Support)}, number, FEW);
    P_C_Rec := RECORD
      Types.t_Discrete class_number; // Used when multiple classifiers being produced at once
      Types.t_Discrete c;            // The class value "C"
      Types.t_FieldReal support;     // Cases count
      Types.t_FieldReal  mu:= 0;     // P(C)
    END;
    // Computing prior probability P(C)
    P_C_Rec pct(ClassCnts le, ClassTots ri) := TRANSFORM
      SELF.class_number := ri.number;
      SELF.c := le.value;
      SELF.support := le.Support;
      SELF.mu := le.Support/ri.TSupport;
    END;
    PC := JOIN(ClassCnts, ClassTots, LEFT.number=RIGHT.number, pct(LEFT,RIGHT), FEW);
    PC_cnt := COUNT(PC);
    // Computing Attributes' mean and variance. mu_c and sigma^2_c.
    AggregatedTriple := RECORD
      Vals.class_number;
      Vals.c;
      Vals.number;
      Types.t_Count support := COUNT(GROUP);
      Types.t_FieldReal mu:=AVE(GROUP, Vals.value);
      Types.t_FieldReal var:= VARIANCE(GROUP, Vals.value);
    END;
    AC:= TABLE(Vals, AggregatedTriple, class_number, c, number);
    Pret := PROJECT(PC, TRANSFORM(BayesResultC, SELF.id := Base + COUNTER, SELF.number := 0, SELF:=LEFT)) +
            PROJECT(AC, TRANSFORM(BayesResultC, SELF.id := Base + COUNTER + PC_cnt, SELF.var:= LEFT.var*LEFT.support/(LEFT.support -1), SELF := LEFT));
    ToField(Pret,o);
    RETURN o;
  END;
  // Transform NumericFiled "mod" to continuos Naive Bayes format model "BayesResultC"
  EXPORT ModelC(DATASET(Types.NumericField) mod) := FUNCTION
    ML.FromField(mod,BayesResultC,o);
    RETURN o;
  END;
  EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep, DATASET(Types.NumericField) mod) := FUNCTION
    dd := DISTRIBUTE(Indep, HASH(id));
    mo := ModelC(mod);
    Inter := RECORD
      Types.t_RecordId    id;
      ClassifierFeatureClass;
      Types.t_FieldReal   value;
      Types.t_FieldReal   likehood:=0; // Probability density P(x=v|c)
    END;
    Inter ProbDensity(dd le, mo ri) := TRANSFORM
      SELF.id := le.id;
      SELF.value:= le.value;
      SELF.likehood := LogScale(exp(-(le.value-ri.mu)*(le.value-ri.mu)/(2*ri.var))/SQRT(2*ML.Utils.Pi*ri.var));
      SELF:= ri;
    END;
    // Likehood or probability density P(x=v|c) is calculated assuming Gaussian distribution of the class based on new instance attribute value and atribute's mean and variance from model
    LogPall := JOIN(dd,mo,LEFT.number=RIGHT.number , ProbDensity(LEFT,RIGHT),MANY LOOKUP);
    // Prior probaility PC
    LogPC:= PROJECT(mo(number=0),TRANSFORM(BayesResultC, SELF.mu:=LogScale(LEFT.mu), SELF:=LEFT));
    post_rec:= RECORD
      LogPall.id;
      LogPall.class_number;
      LogPall.c;
      Types.t_FieldReal prod:= SUM(GROUP, LogPall.likehood);
    END;
    // Likehood and Prior are expressed in LogScale, summing really means multiply
    LikehoodProduct:= TABLE(LogPall, post_rec, class_number, c, id, LOCAL);
    // Posterior probability = prior x likehood_product / evidence
    // We use only the numerator of that fraction, because the denominator is effectively constant.
    // See: http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
    l_result toResult(LikehoodProduct le, LogPC ri) := TRANSFORM
      SELF.id := le.id;               // Instance ID
      SELF.number := le.class_number; // Classifier ID
      SELF.value := ri.c;             // Class value
      SELF.conf:= le.prod + ri.mu;    // Adding mu
    END;
    AllPosterior:= JOIN(LikehoodProduct, LogPC, LEFT.class_number = RIGHT.class_number AND LEFT.c = RIGHT.c, toResult(LEFT, RIGHT), LOOKUP);
    // Normalizing Likehood to deliver Class Probability per instance
    baseExp:= TABLE(AllPosterior, {id, minConf:= MIN(GROUP, conf)},id, LOCAL); // will use this to divide instance's conf by the smallest per id
    l_result toNorm(AllPosterior le, baseExp ri) := TRANSFORM
      SELF.conf:= POWER(2.0, -MIN( le.conf - ri.minConf, 2048));  // minimum probability set to 1/2^2048 = 0 at the end
      SELF:= le;
    END;
    AllOffset:= JOIN(AllPosterior, baseExp, LEFT.id = RIGHT.id, toNorm(LEFT, RIGHT), LOOKUP); // at least one record per id with 1.0 probability before normalization
    gInst := TABLE(AllOffset, {number, id, tot:=SUM(GROUP,conf)}, number, id, LOCAL);
    clDist:= JOIN(AllOffset, gInst,LEFT.number=RIGHT.number AND LEFT.id=RIGHT.id, TRANSFORM(Types.l_result, SELF.conf:=LEFT.conf/RIGHT.tot, SELF:=LEFT), LOCAL);
    RETURN clDist;
  END;
  // Classification function for continuous independent values and model
  EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
    // get class probabilities for each instance
    dClass:= ClassProbDistribC(Indep, mod);
    // select the class with greatest probability for each instance
    sClass := SORT(dClass, id, -conf, LOCAL);
    finalClass:=DEDUP(sClass, id, LOCAL);
    RETURN finalClass;
  END;
END;