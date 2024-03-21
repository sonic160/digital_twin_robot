<template>
  <el-divider content-position="left">
    <h2>Operations</h2>
  </el-divider>
  <div id="top-box">
    <el-button style="" type="primary" plain  @click="saveAll($globalData)">save all data to csv</el-button>
  </div>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 1</h2>
      <el-divider direction="vertical" />
      <el-button id="motor1" type="primary" plain @click="onClick($globalData.v1, $globalData.a1, $globalData.t1, 'motor1')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v1" title="Voltage" YTitle="Volt" color="#FF8000"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t1" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a1" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 2</h2>
      <el-divider direction="vertical" />
      <el-button id="motor2" type="primary" plain @click="onClick($globalData.v2, $globalData.a2, $globalData.t2, 'motor2')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v2" title="Voltage" YTitle="Volt" color="#FF8000"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t2" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a2" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 3</h2>
      <el-divider direction="vertical" />
      <el-button id="motor3" type="primary" plain @click="onClick($globalData.v3, $globalData.a3, $globalData.t3, 'motor3')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v3" title="Voltage" YTitle="Volt" color="#FF8000"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t3" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a3" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 4</h2>
      <el-divider direction="vertical" />
      <el-button id="motor4" type="primary" plain @click="onClick($globalData.v4, $globalData.a4, $globalData.t4, 'motor4')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v4" title="Voltage" YTitle="Volt" color="#FF8000"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t4" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a4" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 5</h2>
      <el-divider direction="vertical" />
      <el-button id="motor5" type="primary" plain @click="onClick($globalData.v5, $globalData.a5, $globalData.t5, 'motor5')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v5" title="Voltage" YTitle="Volt" color="#FF8000"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t5" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a5" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
  <el-divider content-position="left">
    <span style="display: flex; justify-content: center; align-items: center;">
      <h2>Motor 6</h2>
      <el-divider direction="vertical" />
      <el-button id="motor6" type="primary" plain @click="onClick($globalData.v6, $globalData.a6, $globalData.t6, 'motor6')"><el-icon><Download /></el-icon></el-button>
    </span>
  </el-divider>
  <el-row>
    <el-col :span="8">
      <LineChart :data="$globalData.v6" title="Voltage" YTitle="Volt" color="#FF8000"  />
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.t6" title="Temprature" YTitle="Degrees Centigrade"/>
    </el-col>
    <el-col :span="8">
      <LineChart :data="$globalData.a6" title="Position" YTitle="Degree" color="#CC0066"/>
    </el-col>
  </el-row>
</template>

<script setup>
import LineChart from '../components/LineChart.vue';

function genCSV(csv, name) {
  
  const blob = new Blob([csv], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);

  const tbutton = document.createElement("a");
  tbutton.style.display = "none";
  tbutton.href = url;
  tbutton.download = `merged_data_${name}.csv`;
  document.body.appendChild(tbutton);
  tbutton.click();
  document.body.removeChild(tbutton);
}

function onClick(v, a, t, buttonId) {
  let csv = "time, v, a, t\n";
  const times = new Set([...v.map(item => item.label), ...a.map(item => item.label), ...t.map(item => item.label)]);
  times.forEach(time => {
    const vData = v.find(item => item.label === time);
    const aData = a.find(item => item.label === time);
    const tData = t.find(item => item.label === time);
    csv += `${time},${vData ? vData.y : ""},${aData ? aData.y : ""},${tData ? tData.y : ""}\n`;
  });
  genCSV(csv, buttonId);
}

function saveAll(globalData) {
  let csv = "time, motor, v, a, t\n";
  const times = new Set([
    ...globalData.v1.map(item => item.label), ...globalData.a1.map(item => item.label), ...globalData.t1.map(item => item.label),
    ...globalData.v2.map(item => item.label), ...globalData.a2.map(item => item.label), ...globalData.t2.map(item => item.label),
    ...globalData.v3.map(item => item.label), ...globalData.a3.map(item => item.label), ...globalData.t3.map(item => item.label),
    ...globalData.v4.map(item => item.label), ...globalData.a4.map(item => item.label), ...globalData.t4.map(item => item.label),
    ...globalData.v5.map(item => item.label), ...globalData.a5.map(item => item.label), ...globalData.t5.map(item => item.label),
    ...globalData.v6.map(item => item.label), ...globalData.a6.map(item => item.label), ...globalData.t6.map(item => item.label)
  ]);
  for (let i = 1; i <= 6; i++) {
    const v = `v${i}`;
    const a = `a${i}`;
    const t = `t${i}`;
    times.forEach(time => {
      const vData = globalData[v].find(item => item.label === time);
      const aData = globalData[a].find(item => item.label === time);
      const tData = globalData[t].find(item => item.label === time);
      console.log(time, `motor${i}`, vData, aData, tData);
      csv += `${time},motor${i},${vData ? vData.y : ""},${aData ? aData.y : ""},${tData ? tData.y : ""}\n`;
    });
  }
  genCSV(csv, "6motors");
}
</script>

<style scoped>
  #top-box {
    height: 48px;
    margin-bottom: 20px;
    padding: 0 25px;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    z-index: 999;
  }
</style>