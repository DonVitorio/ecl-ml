IMPORT * FROM ML;
IMPORT ML.Tests.Explanatory as TE;
IMPORT * FROM ML.Lazy;

Depth:= 10;
MedianDepth:= 15;
/*
weatherRecord := RECORD
	Types.t_RecordID id;
	Types.t_FieldNumber outlook;
	Types.t_FieldNumber temperature;
	Types.t_FieldNumber humidity;
	Types.t_FieldNumber windy;
	Types.t_FieldNumber play;
END;
weather_Data := DATASET([
{1,0,0,1,0,0},
{2,0,0,1,1,0},
{3,1,0,1,0,1},
{4,2,1,1,0,1},
{5,2,2,0,0,1},
{6,2,2,0,1,0},
{7,1,2,0,1,1},
{8,0,1,1,0,0},
{9,0,2,0,0,1},
{10,2,1,0,0,1},
{11,0,1,0,1,1},
{12,1,1,1,1,1},
{13,1,0,0,0,1},
{14,2,1,1,1,0}],
weatherRecord);
weather_Data;
ToField(weather_Data, full_ds);
distribDS := DISTRIBUTE(full_ds, HASH(id));
indepData:= distribDS(number<5);
depData:= ML.Discretize.ByRounding(distribDS(number=5));

newdata:= DATASET([
{6,2,2,0,1,0},
{7,1,2,0,1,1},
{8,0,1,1,0,0},
{9,0,2,0,0,1},
{10,2,1,0,0,1}], weatherRecord);
newData;
ToField(newdata, full_qpdata);
qpdata:= DISTRIBUTE(full_qpdata(number<5), HASH(id));
*/
indep_data:= TABLE(TE.MonkDS.Train_Data,{id, a1, a2, a3, a4, a5, a6});
dep_data:= TABLE(TE.MonkDS.Train_Data,{id, class});
ToField(indep_data, indepData);
ToField(dep_data, pr_dep);
depData := ML.Discretize.ByRounding(pr_dep);

indep_test:= TABLE(TE.MonkDS.Test_Data,{id, a1, a2, a3, a4, a5, a6});
dep_test:= TABLE(TE.MonkDS.Test_Data,{id, class});
ToField(indep_test, IndepTest);
ToField(dep_test, pr_depT);
depTest := ML.Discretize.ByRounding(pr_depT);


iknn:= Lazy.KNN(3);
ikdt:= iknn.KDTreeNNSearch(); // Using default values Depth=10, MedianDepth=0

computed:=  ikdt.ClassifyC(IndepData, depData, IndepTest);
comp:= ikdt.Compare(depTest, computed);
computed;
comp.Raw;
comp.CrossAssignments;
comp.PrecisionByClass;
comp.Headline;

TestModule:=  ikdt.TestC(IndepData, depData);
TestModule.Raw;
TestModule.CrossAssignments;
TestModule.PrecisionByClass;
TestModule.Headline;


















