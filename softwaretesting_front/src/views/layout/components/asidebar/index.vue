<template>
  <el-menu
    class="el-menu-vertical-demo"
    @open="handleOpen"
    @close="handleClose"
    background-color="#f6f9ff"
    text-color="#012970"
    active-text-color="#4154f1"
    :collapse="isCollapse"
    :default-openeds="openeds"
    router
  >
    <el-menu-item index="/">
      <template slot="title">
        <i class="el-icon-star-on"></i>
        <span>首页</span>
      </template>
    </el-menu-item>

    <el-submenu index="1">
      <template slot="title">
        <i class="el-icon-star-on"></i>
        <span>第一部分 课程练习</span>
      </template>
      <el-menu-item-group>
        <el-menu-item index="/triangle"
          >(程序题)第1、9题:判断三角形</el-menu-item
        >
        <el-menu-item index="/calendar"
          >(程序题)第2、10、11题:万年历问题</el-menu-item
        >
        
        <el-menu-item index="/sales">(程序题)第4题:电脑销售系统</el-menu-item>
        <el-menu-item index="/cash">(程序题)第7题:电信收费问题</el-menu-item>
        <el-menu-item index="/atm">(程序题)第12、15题:ATM</el-menu-item>
        <el-menu-item index="/sale">(程序题)第17题:销售系统</el-menu-item>
        <el-menu-item index="/three">第3题:销售管理系统</el-menu-item>
        <el-menu-item index="/five">第5题:电商平台测试</el-menu-item>
        <el-menu-item index="/six">第6题:项目测试用例</el-menu-item>
        <el-menu-item index="/eight">第8题:实时系统</el-menu-item>
        <el-menu-item index="/thirteen">第13题:ERP系统</el-menu-item>
        <el-menu-item index="/fourteen">第14题:实时Web系统</el-menu-item>
        <!-- <el-menu-item index="/nine">第15题:ATM机</el-menu-item> -->
        <el-menu-item index="/sixteen">第16题:构建程序图</el-menu-item>
      </el-menu-item-group>
    </el-submenu>
    <el-submenu index="2">
      <template slot="title">
        <i class="el-icon-star-on"></i>
        <span>第二部分 期末项目演示</span>
      </template>

  
        <el-menu-item-group>
          <el-menu-item index="/lab">多模态故障检测实验</el-menu-item>
        </el-menu-item-group>
      </el-submenu>
    
  </el-menu>
  
</template>

<script>
export default {
  name: "AsideBar",
  components: {},
  props: ["is-collapse"],
  data() {
    return {
      openeds: ["1"],
    };
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {
    handleOpen(key, keyPath) {
      console.log(key, keyPath);
    },
    handleClose(key, keyPath) {
      console.log(key, keyPath);
    },
  },
};
</script>

<style scoped lang="less">
.el-menu-vertical-demo:not(.el-menu--collapse) {
  width: 300px;
  min-height: 400px;
}
.home {
  background-color: #1a202c;
}
</style>


