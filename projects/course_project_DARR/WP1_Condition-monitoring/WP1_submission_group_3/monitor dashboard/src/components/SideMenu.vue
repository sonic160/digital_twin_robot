<template>
    <el-menu
        active-text-color="#ffd04b"
        background-color="#545c64"
        class="el-menu-vertical-demo"
        default-active="1"
        text-color="#fff"
        :collapse="isCollapse"
    >
        <div id="menu-div">
            <div>
                <router-link to="/">
                    <el-menu-item index="1">
                        <el-icon><TrendCharts /></el-icon>
                        <template #title>Home</template>
                    </el-menu-item>
                </router-link>
                <router-link to="/table">
                    <el-menu-item index="2">
                        <el-icon><Grid /></el-icon>
                        <template #title>Table</template>
                    </el-menu-item>
                </router-link>
            </div>
            <el-menu-item @click="onClick">
                <el-icon v-if="isCollapse"><ArrowRight /></el-icon>
                <el-icon v-else><ArrowLeft /></el-icon>
                <template #title>{{ buttonName }}</template>
            </el-menu-item>
        </div>
    </el-menu>
</template>

<script setup>
import {ref} from 'vue'

const isCollapse = ref(true)
const buttonName = ref("展开菜单")

function onClick() {
    isCollapse.value = !isCollapse.value
    if (isCollapse.value) {
        buttonName.value = "展开菜单"
    } else {
        buttonName.value = "隐藏菜单"
    }
}

</script>

<style scoped>
    .el-menu {
        background-image: linear-gradient(#60596b, #322c2a);
    }

    #menu-div {
        height: 100%;
        display: flex;
        justify-content: space-between;
        flex-direction: column;
    }
</style>