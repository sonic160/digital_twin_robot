import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import ECharts from 'vue-echarts' 
import CanvasJSChart from '@canvasjs/vue-charts'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

const app = createApp(App)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}

const globalData = {
    v1: [],
    v2: [],
    v3: [],
    v4: [],
    v5: [],
    v6: [],
    t1: [],
    t2: [],
    t3: [],
    t4: [],
    t5: [],
    t6: [],
    a1: [],
    a2: [],
    a3: [],
    a4: [],
    a5: [],
    a6: [],
};

app.config.globalProperties.$globalData = globalData;

var ws = new WebSocket("ws://localhost:45709");
ws.onmessage = function(event) {
  const tdata = JSON.parse(event.data);
  const data = tdata.data;
  const time = tdata.time;
  globalData.v1.push({'label': time, 'y': data.motor1.v});
  globalData.t1.push({'label': time, 'y': data.motor1.t});
  globalData.a1.push({'label': time, 'y': data.motor1.a});
  globalData.v2.push({'label': time, 'y': data.motor2.v});
  globalData.t2.push({'label': time, 'y': data.motor2.t});
  globalData.a2.push({'label': time, 'y': data.motor2.a});
  globalData.v3.push({'label': time, 'y': data.motor3.v});
  globalData.t3.push({'label': time, 'y': data.motor3.t});
  globalData.a3.push({'label': time, 'y': data.motor3.a});
  globalData.v4.push({'label': time, 'y': data.motor4.v});
  globalData.t4.push({'label': time, 'y': data.motor4.t});
  globalData.a4.push({'label': time, 'y': data.motor4.a});
  globalData.v5.push({'label': time, 'y': data.motor5.v});
  globalData.t5.push({'label': time, 'y': data.motor5.t});
  globalData.a5.push({'label': time, 'y': data.motor5.a});
  globalData.v6.push({'label': time, 'y': data.motor6.v});
  globalData.t6.push({'label': time, 'y': data.motor6.t});
  globalData.a6.push({'label': time, 'y': data.motor6.a});
};


app.use(router)
app.use(CanvasJSChart)
app.use(ElementPlus)
app.component('v-chart', ECharts)
app.mount('#app')