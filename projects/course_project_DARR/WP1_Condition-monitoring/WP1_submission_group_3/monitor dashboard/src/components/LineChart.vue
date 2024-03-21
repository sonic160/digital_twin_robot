<template>
  <CanvasJSChart :options="options" :styles="styleOptions" @chart-ref="chartInstance"/>
</template>                   

<script setup>
  import { defineProps, ref } from 'vue';
  
  const props = defineProps({
    "title": {
      "type": String,
      "default": "LineChart"
    },
    "data": {
      "type": Array[{ 
        "label": String,
        "y": Number
      }],
      // "default": [{'label': '01:00', 'y': 3}, {'label': '01:01', 'y': 4}, {'label': '01:02', 'y': 4}, {'label': '01:03', 'y': 5}, {'label': '01:04', 'y': 4}, {'label': '01:05', 'y': 4}, {'label': '01:06', 'y': 5}, {'label': '01:07', 'y': 4}, {'label': '01:08', 'y': 3}, {'label': '01:09', 'y': 3}, {'label': '01:10', 'y': 4}, {'label': '01:11', 'y': 5}, {'label': '01:12', 'y': 3}, {'label': '01:13', 'y': 5}, {'label': '01:14', 'y': 3}, {'label': '01:15', 'y': 5}, {'label': '01:16', 'y': 3}, {'label': '01:17', 'y': 3}, {'label': '01:18', 'y': 3}, {'label': '01:19', 'y': 4}, {'label': '01:20', 'y': 4}, {'label': '01:21', 'y': 4}, {'label': '01:22', 'y': 4}, {'label': '01:23', 'y': 5}, {'label': '01:24', 'y': 3}, {'label': '01:25', 'y': 4}, {'label': '01:26', 'y': 4}, {'label': '01:27', 'y': 4}, {'label': '01:28', 'y': 3}, {'label': '01:29', 'y': 4}, {'label': '01:30', 'y': 5}, {'label': '01:31', 'y': 3}, {'label': '01:32', 'y': 5}, {'label': '01:33', 'y': 4}, {'label': '01:34', 'y': 4}, {'label': '01:35', 'y': 4}, {'label': '01:36', 'y': 5}, {'label': '01:37', 'y': 4}, {'label': '01:38', 'y': 5}, {'label': '01:39', 'y': 5}, {'label': '01:40', 'y': 4}, {'label': '01:41', 'y': 3}, {'label': '01:42', 'y': 5}, {'label': '01:43', 'y': 4}, {'label': '01:44', 'y': 4}, {'label': '01:45', 'y': 3}, {'label': '01:46', 'y': 4}, {'label': '01:47', 'y': 5}, {'label': '01:48', 'y': 3}, {'label': '01:49', 'y': 4}, {'label': '01:50', 'y': 3}, {'label': '01:51', 'y': 5}, {'label': '01:52', 'y': 3}, {'label': '01:53', 'y': 4}, {'label': '01:54', 'y': 4}, {'label': '01:55', 'y': 4}, {'label': '01:56', 'y': 4}, {'label': '01:57', 'y': 4}, {'label': '01:58', 'y': 4}, {'label': '01:59', 'y': 5}]
    },
    "XTitle": {
      "type": String,
      "default": "Time"
    },
    "YTitle": {
      "type": String,
      "default": "YTitle"
    },
    "color": {
      "type": String,
      "default": "#0066cc"
    }
  })

  const chart = ref(null);
 
  const options = ref({
    theme: "light2",
    animationEnabled: true,
    exportEnabled: true,
    title: {
      text: props.title
    },
    axisY: {
      title: props.YTitle
    },
    axisX: {
      title: props.XTitle
    },
    data: [{
      type: "spline",
      color: props.color,
      dataPoints: props.data
    }]
  })

  const styleOptions = {
    width: "100%",
    height: "200px"
  }

  const dataPoints = ref([])

  function updateChart(count) {
    // dataPoints.value = props.data.slice(-count)
    dataPoints.value = props.data
    options.value = {
      theme: "light2",
      zoomEnabled: true,
      animationEnabled: true,
      exportEnabled: true,
      title: {
        text: props.title
      },
      axisY: {
        title: props.YTitle
      },
      axisX: {
        title: props.XTitle
      },
      data: [{
        type: "spline",
        color: props.color,
        dataPoints: dataPoints
      }]
    }
    setTimeout(function() {updateChart(10)}, 100);
  }

  function chartInstance(c) {
    chart.value = c;
    updateChart(10)
  }

  updateChart()
</script>
