<script setup>
import { RouterView } from 'vue-router'
import SideMenu from './components/SideMenu.vue';
</script>

<template>
  <div id="base-layout">
    <el-container>
      <SideMenu />
      <el-container>
        <el-header>
          <div class="header-div">
            <el-icon style="padding-right: 10px;"><Odometer /></el-icon>
            <h1>Dashboard</h1>
            
          </div>
          <div class="header-div">
            <span>Risk Reliability Resilience (R3) Research Group</span>
            <img style="height: 40px; width: 40px; padding: 10px;" src="@/assets/logo.svg"/>
            <el-button type="danger" plain><el-icon><Bell /></el-icon></el-button>
          </div>
        </el-header>
        <el-main>
          <RouterView />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<style scoped>
  .el-container, #base-layout {
    height: 100%;
  }
  .SideMenu {
    width: 200px;
    height: 100%;
    padding: 0;
    margin: 0;
  }
  .el-header, .header-div {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }

  .el-header {
    border-bottom: 2px solid gray;
  }
</style>