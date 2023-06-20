<template>
  <div class="single-case">
    <el-form
      class="single-form"
      :label-position="labelPosition"
      label-width="50px"
      :model="formLabelAlign"
    >
      <el-form-item>
        <div class="model-row">
          <span class="model-label">选择MODEL:</span>
          <div class="button-row">
            <el-radio-group v-model="formLabelAlign.model">
              <el-radio label="PU" size="large" border>PU</el-radio>
              <el-radio label="CNN" size="large" border>CNN</el-radio>
              <el-radio label="Classify" size="large" border>Classify</el-radio
              >a
              <el-radio label="DTL" size="large" border>DTL</el-radio>
            </el-radio-group>
          </div>
        </div>
      </el-form-item>

      <el-form-item>
        <div class="model-row">
          <span class="model-label">选择TRAIN的数据:</span>
          <el-select
            v-model="formLabelAlign.train_data"
            class="select-train"
            placeholder="Select"
            size="large"
          >
            <el-option
              v-for="item in options1"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
      </el-form-item>

      <el-form-item >
        <div class="model-row">
          <span class="model-label">选择TEST的数据:</span>
          <!-- <el-input v-model="formLabelAlign.test"></el-input> -->
          <el-select
            v-model="formLabelAlign.data"
            class="select-test"
            placeholder="Select"
            size="large"
          >
            <el-option
              v-for="item in options2"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
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
      <!-- <span>实际输出：{{ actual }}</span>
      <el-divider direction="vertical"></el-divider>
      <span>运行信息：{{ info }}</span>
      <el-divider direction="vertical"></el-divider>
       -->

      <el-table class="table2" :data="resultData">
        <el-table-column prop="P" label="P"></el-table-column>
        <el-table-column prop="R" label="R"></el-table-column>
        <el-table-column prop="F1" label="F1"></el-table-column>
        <!-- 添加更多列定义 -->
      </el-table>
    </div>
  </div>
</template>

<script>
import { testlabsingle } from "@/api/labtest.js";
import { ref } from "vue";
export default {
  name: "SingleCase",
  components: {},
  props: {},

  data() {
    return {
      actual: "",
      info: "",
      labelPosition: "right",
      tableHeader: '输出结果', 
      formLabelAlign: {
        model: ref("PU"),
        train: ref(""),
        test: ref(""),
      },
      date: "",
      loading: false,
      resultData: [],
      options1: [
        {
          value: "r",
          label: "数据集r",
        },
        {
          value: "rt",
          label: "数据集rt",
        },
        {
          value: "rtd",
          label: "数据集rtd",
        },
      ],
      options2: [
        {
          value: "r-s",
          label: "数据集r-s",
        },
        {
          value: "r-m",
          label: "数据集r-m",
        },
        {
          value: "r-l",
          label: "数据集r-l",
        },
        {
          value: "rt-s",
          label: "数据集rt-s",
        },
        {
          value: "rt-m",
          label: "数据集rt-m",
        },
        {
          value: "rt-l",
          label: "数据集rt-l",
        },
        {
          value: "rtd-s",
          label: "数据集rtd-s",
        },
        {
          value: "rtd-m",
          label: "数据集rtd-m",
        },
        {
          value: "rtd-l",
          label: "数据集rtd-l",
        },
      ],
    };
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {
    doTest() {
      const newData = {};
      const columnMapping = {
        model: "model",
        train_data: "train_data",
        data: "data",
      };
      this.loading = true;

      // 将 formLabelAlign 中的属性映射到新的 newData 对象中
      for (const oldColumn in columnMapping) {
        const newColumn = columnMapping[oldColumn];
        newData[newColumn] = this.formLabelAlign[oldColumn];
      }

      console.log(newData);

      // let formdata = {
      //   id: "TS1",
      //   year: this.formLabelAlign.year,
      //   month: this.formLabelAlign.month,
      //   day: this.formLabelAlign.day,
      //   expectation: this.formLabelAlign.expectation,
      // };

      // let data = {
      //   calendar_test_list: [formdata],
      // };

      testlabsingle(newData)
        .then((res) => {
          console.log(res);
          this.resultData = [
            {
              P: res.data.P,
              R: res.data.R,
              F1: res.data.F1,
            },
          ];

          this.$message({
            message: "测试成功",
            type: "success",
          });
          _this.loading = false;
        })
        .catch((err) => {
          _this.$message.error("Server Error");
          _this.loading = false;
        });
    },
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
  width: 100%;
}
.box-card {
  padding: 0;
}
.centered-forms {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 20px 0;
}
.single-case {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 20px 0;
}
.table2{
  margin-top: 50px;
}
.button-row {
  display: flex;
}
.model-row {
  display: flex;
  align-items: center;
}

.model-label {
  margin-right: 10px;
  margin-left: -100px;
}
.el-select{
   display: flex;
  align-items: center;
  margin-left: 50px;

}
.result{
 display: flex;
 margin-top: 50px;
}
.el-radio-group {
  display: flex;
  align-items: center;
  margin-left: 50px;
}
.model-item .el-form-item__label {
  margin-right: 20px;
}
.select-train{
  margin-left: 140px;
}
.select-test{
  margin-left: 150px;
}
</style>
