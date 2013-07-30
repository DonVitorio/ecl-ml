IMPORT ML;
IMPORT * FROM ML.Types;
IMPORT * FROM ML.Trees;

weatherRecord := RECORD
	Types.t_RecordID id;
	Types.t_FieldReal outlook;
	Types.t_FieldReal temperature;
	Types.t_FieldReal humidity;
	Types.t_FieldReal windy;
	Types.t_Discrete play;
END;
weather_Data := DATASET([
{1,0,0,1,0,0}, //0
{2,0,0,1,1,0}, //0
{3,1,0,1,0,1}, //1
{4,2,1,1,0,1}, //1
{5,2,2,0,0,1}, //1
{6,2,2,0,1,0}, //0
{7,1,2,0,1,1}, //1
{8,0,1,1,0,0}, //0
{9,0,2,0,0,1}, //1
{10,2,1,0,0,1},//1 
{11,0,1,0,1,1},//1
{12,1,1,1,1,1},//1
{13,1,0,0,0,1},//1
{14,2,1,1,1,0}],//0
weatherRecord);
OUTPUT(weather_Data, NAMED('weather_Data'));
ML.ToField(weather_Data, full_ds);
indepData:= full_ds(number<5);
depData:= full_ds(number=5);

ind0 := ML.Utils.Fat(indepData); // Ensure no sparsity in independents
cNode init(ind0 le, depData ri) := TRANSFORM
  SELF.node_id := 1;
  SELF.level := 1;
  SELF.depend := ri.value;	// Actually copies the dependant value to EVERY node - paying memory to avoid downstream cycles
  SELF := le;
END;
p_level :=1;
res0 := JOIN(ind0, depData, LEFT.id = RIGHT.id, init(LEFT,RIGHT));
OUTPUT(res0);
res1:= ML.Trees.BinaryPartitionC(res0, 1, 1);
OUTPUT(res1, ALL);

res2:= ML.Trees.BinaryPartitionC(res1, 2, 1);
OUTPUT(res2, ALL);

res3:= ML.Trees.BinaryPartitionC(res2, 3, 1);
OUTPUT(res3, ALL);

res4:= ML.Trees.BinaryPartitionC(res3, 4, 1);
OUTPUT(res4, ALL);

res5:= ML.Trees.BinaryPartitionC(res4, 5, 1);
OUTPUT(res5, ALL);


