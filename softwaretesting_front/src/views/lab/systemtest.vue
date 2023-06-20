<template>
  <div class="system-test">
    <div class="main-header">
      <div class="button-group">
        <el-button
          class="main-button"
          type="success"
          plain
          @click="doTest"
          :loading="loading"
          >进行测试<i class="el-icon-upload el-icon--right"></i
        ></el-button>
        <el-button
          @click="reset(value)"
          class="reset-button"
          type="warning"
          plain
          >重置</el-button
        >
      </div>
    </div>

    <el-divider content-position="right">测试用例</el-divider>

    <div class="main-table">
      <el-table
        :data="tableData"
        :height="tableHeight"
        border
        style="width: 100%"
        v-loading="loading"
        :row-class-name="tableRowClassName"
      >
        <el-table-column
          label="测试用例编号"
          width="120"
          align="center"
          type="index"
        >
          <template slot-scope="scope">
            <span>{{ scope.$index + 1 }}</span>
          </template>
        </el-table-column>
        <el-table-column label="输入数据" align="center">
          <el-table-column
            prop="model"
            label="model"
            width="120"
            align="center"
          ></el-table-column>
          <el-table-column
            prop="train"
            width="120"
            label="train"
            align="center"
          ></el-table-column>
          <el-table-column
            prop="test"
            width="120"
            label="test"
            align="center"
          ></el-table-column>
        </el-table-column>

        <el-table-column label="实际输出" align="center">
          <el-table-column
            prop="precision_real"
            label="precision"
            align="center"
          ></el-table-column>
          <el-table-column
            prop="recall_real"
            label="recall"
            align="center"
          ></el-table-column>
          <el-table-column
            prop="f1score_real"
            label="f1-score"
            align="center"
          ></el-table-column>
        </el-table-column>
        <!-- <el-table-column
          prop="info"
          label="程序运行信息"
          align="center"
        ></el-table-column> -->

        <!-- <el-table-column prop="state" label="测试结果" align="center">
          <template slot-scope="scope">
            <div v-if="scope.row.state == true" class="icon-svg">
              <i class="el-icon-check"></i><span>测试通过</span>
            </div>
            <div v-if="scope.row.state == false" class="icon-svg">
              <i class="el-icon-close"></i><span>测试未通过</span>
            </div>
          </template>
        </el-table-column> -->
        <el-table-column
          prop="time"
          label="测试时间"
          align="center"
        ></el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script>
import mock_1_json from "@/mock/lab/lab_mock_1.json";
import mock_2_json from "@/mock/lab/lab_mock_2.json";

// import mock_3_json from "@/mock/lab/lab_mock_3.json";
import { testlab } from "@/api/labtest.js";
export default {
  name: "SystemTest",
  components: {},
  props: ["parentHeight"],
  data() {
    return {
      tableData: [
     
        {  model: "DTL",   train:  "r" ,     test: "r-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "r-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "r-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "r" ,     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "r-s" ,        precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rt",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "r-l" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "rt-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "rt-m" ,      precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rt",     test: "rt-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rt",     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rt",     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",    test:  "r-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rtd",     test: "r-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rtd",     test: "rtd-s" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "DTL",    train:  "rtd",     test: "rtd-m" ,    precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "DTL",    train:  "rtd",     test: "rtd-l" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},

         {  model:"CNN",   train:  "r" ,     test: "r-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "CNN",    train:  "r" ,     test: "r-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "CNN",    train:  "r" ,     test: "r-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "CNN",    train:  "r" ,     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "r" ,     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "r" ,     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "r" ,     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "r" ,     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "r" ,     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "r-s" ,        precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rt",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "r-l" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "rt-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "rt-m" ,      precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rt",     test: "rt-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rt",     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rt",     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",    test:  "r-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rtd",     test: "r-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rtd",     test: "rtd-s" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "CNN",    train:  "rtd",     test: "rtd-m" ,    precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "CNN",    train:  "rtd",     test: "rtd-l" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},

         { model: "Classify",   train:  "r" ,     test: "r-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "r-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "r-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {   model: "Classify",    train:  "r" ,     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "r-s" ,        precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rt",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "r-l" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "rt-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "rt-m" ,      precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rt",     test: "rt-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rt",     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rt",     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",    test:  "r-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rtd",     test: "r-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rtd",     test: "rtd-s" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "Classify",    train:  "rtd",     test: "rtd-m" ,    precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "Classify",    train:  "rtd",     test: "rtd-l" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},

         {  model: "PU",   train:  "r" ,     test: "r-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "r-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "r-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "r" ,     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "r-s" ,        precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rt",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "r-l" ,       precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "rt-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "rt-m" ,      precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rt",     test: "rt-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "rtd-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rt",     test: "rtd-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rt",     test: "rtd-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",    test:  "r-s" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",     test: "r-m" ,       precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rtd",     test: "r-l" ,      precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",     test: "rt-s" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",     test: "rt-m" ,     precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",     test: "rt-l" ,     precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rtd",     test: "rtd-s" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},
        {  model: "PU",    train:  "rtd",     test: "rtd-m" ,    precision_real:"",recall_real:"",f1score_real:"",time:"" },
        {  model: "PU",    train:  "rtd",     test: "rtd-l" ,    precision_real:"",recall_real:"",f1score_real:"",time:""},
      ],
      loading: false,
      classState: [],
      testTime: null,
    };
  },
  computed: {
    tableHeight() {
      return this.parentHeight - 260 > 650 ? 650 : this.parentHeight - 260;
    },
  },
  watch: {
    value: {
      handler(newVal) {
        this.reset(newVal);
      },
      immediate: false,
    },
  },
  created() {},
  mounted() {
    this.initTableData();
  },
  methods: {
    //初始化
    initTableData() {
      tableData["precision_real"] = "";
      tableData["recall_real"] = "";
      tableData["f1score_real"] = "";
      tableData["time"] = "";
      console.log(tableData);
    },
    tableRowClassName({ row, rowIndex }) {
      return this.classState[rowIndex];
    },
    doTest() {
      let newData = {};
      const columnMapping = {
        model: "model",
        train: "train_data",
        test: "data",
      };
      const _this = this;
      this.loading = true;

      newData = this.tableData.map((item) => {
        const newItem = {};
        for (const oldColumn in columnMapping) {
          const newColumn = columnMapping[oldColumn];
          newItem[newColumn] = item[oldColumn];
        }
        return newItem;
      });
      console.log(newData);

      // 记录开始时间
      const startTime = new Date();

      testlab(newData)
        .then((res) => {
          console.log(res);
          // 计算测试时间
          const endTime = new Date();
           this.tableData.time = (endTime - startTime) / 1000; // 转换为秒
          _this.tableData.forEach((item, index) => {
            let responseObject = res.data.test_result[index];
            item.precision_real = responseObject.P;
            item.recall_real = responseObject.R;
            item.f1score_real = responseObject.F1;
            // item.state =
            //   responseObject.test_result == "测试通过" ? true : false;
            // _this.classState[index] = item["state"]
            //   ? "success-row"
            //   : "error-row";
          });
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
    reset(value) {
      this.initTableData(mock_1_json);
    },
  },
};
</script>

<style scoped lang="less">
/deep/ .el-table .error-row {
  background: #fff0f0;
}
/deep/ .el-table .success-row {
  background-color: #f7fff9;
}
.main-button {
  width: 500px;
  margin-top: 10px;
}
.reset-button {
  width: 200px;
  margin-top: 10px;
}
.main-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.main-table {
  height: 100%;
  display: flex;
  align-items: center;
}
</style>
