IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT ML.Mat;
IMPORT ML.Tests.Explanatory as TE;

ML.AppendID(ML.Tests.Explanatory.IrisDS, id, dOrig);
ML.ToField(dOrig, dMatrix);
originalData := dMatrix;
Types.t_Discrete pctSize := 70;

sampleSplits:= ML.Sampling.RndSampleSplitNum(originalData, 70);
lds:= sampleSplits.LeftSplit;
rds:= sampleSplits.RightSplit;
OUTPUT(lds, ALL);
OUTPUT(rds, ALL);
