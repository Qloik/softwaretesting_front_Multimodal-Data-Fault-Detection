<template>
  <div class="single-case">
    <el-form
      class="single-form"
      :label-position="labelPosition"
      label-width="400px"
      :model="formLabelAlign"
    >
      <el-form-item label="年销售额">
        <el-input v-model="formLabelAlign.sales"></el-input>
      </el-form-item>
      <el-form-item label="请假天数">
        <el-input v-model="formLabelAlign.days"></el-input>
      </el-form-item>
      <el-form-item label="现金到账">
        <el-input v-model="formLabelAlign.money"></el-input>
      </el-form-item>
      <el-form-item label="佣金系数预期输出">
        <el-input v-model="formLabelAlign.expectation"></el-input>
      </el-form-item>
    </el-form>
          <el-button
        class="main-button"
        type="success"
        plain
        @click="doTest"
        :loading="loading"
        >进行测试<i class="el-icon-upload el-icon--right"></i
      ></el-button>
        <div>
    <span>佣金系数实际输出：{{actual}}</span>
    <el-divider direction="vertical"></el-divider>
    <span>运行信息：{{info}}</span>
    <el-divider direction="vertical"></el-divider>
  </div>
  </div>
</template>

<script>

import { testmoney } from "@/api/moneytest.js";

export default {
  name: "SingleCase",
  components: {},
  props: {},
  data() {
    return {
      actual:"",
      info:"",
      labelPosition: 'right',
        formLabelAlign: {
          sales: 0,
          days: 0,
          money: 0,
          expectation:0,
        }, 
        date:"",
        loading:false,

    };
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {
    doTest(){
      let formdata = {
        id: "TS1",
        sales: this.formLabelAlign.sales,
        days: this.formLabelAlign.days,
        money:this.formLabelAlign.money,
        expectation: this.formLabelAlign.expectation,
      }
      let data = {
        money_test_list:[formdata],
      }
      testmoney(data).then((res)=>{
        this.actual = res.data.test_result[0].actual;
        this.info = res.data.test_result[0].info;
      })

    }
  },
};
</script>

<style scoped>
.item {
  margin-bottom: 10px;
}
.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}
.clearfix:after {
  clear: both;
}
.main-form {
  margin-top: 10px;
}
.main-button {
  width:100%;

}
.box-card {
  padding: 0;
}
.single-form{
  width:600px;
  top:50%;
  left:50%;
}
</style>
