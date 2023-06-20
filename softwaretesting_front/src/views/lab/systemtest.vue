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
          ><template slot-scope="scope">
    <span>{{ formatPercentage(scope.row.precision_real) }}</span>
  </template></el-table-column>
          <el-table-column
            prop="recall_real"
            label="recall"
            align="center"
          ><template slot-scope="scope">
    <span>{{ formatPercentage(scope.row.precision_real) }}</span>
  </template></el-table-column>
          <el-table-column
            prop="f1score_real"
            label="f1-score"
            align="center"
          ><template slot-scope="scope">
    <span>{{ formatPercentage(scope.row.precision_real) }}</span>
  </template></el-table-column>
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
        <!-- <el-table-column
          prop="time"
          label="测试时间"
          align="center"
        ></el-table-column> -->
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
      tableData:[
    {
        "model": "DTL",
        "train": "r",
        "test": "r-s",
        "precision_real": 9.76,
        "recall_real": 100.0,
        "f1score_real": 17.78,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "r-m",
        "precision_real": 95.0,
        "recall_real": 79.17,
        "f1score_real": 86.36,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "r-l",
        "precision_real": 96.55,
        "recall_real": 100.0,
        "f1score_real": 98.25,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "rt-m",
        "precision_real": 21.43,
        "recall_real": 100.0,
        "f1score_real": 35.29,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "rt-l",
        "precision_real": 93.94,
        "recall_real": 100.0,
        "f1score_real": 96.88,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "rtd-m",
        "precision_real": 27.06,
        "recall_real": 100.0,
        "f1score_real": 42.59,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "r",
        "test": "rtd-l",
        "precision_real": 83.51,
        "recall_real": 100.0,
        "f1score_real": 91.01,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "r-s",
        "precision_real": 9.76,
        "recall_real": 100.0,
        "f1score_real": 17.78,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "r-m",
        "precision_real": 28.24,
        "recall_real": 100.0,
        "f1score_real": 44.04,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "r-l",
        "precision_real": 32.56,
        "recall_real": 100.0,
        "f1score_real": 49.12,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "rt-m",
        "precision_real": 94.12,
        "recall_real": 88.89,
        "f1score_real": 91.43,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "rt-l",
        "precision_real": 98.94,
        "recall_real": 100.0,
        "f1score_real": 99.47,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "rtd-m",
        "precision_real": 27.71,
        "recall_real": 100.0,
        "f1score_real": 43.4,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rt",
        "test": "rtd-l",
        "precision_real": 83.51,
        "recall_real": 100.0,
        "f1score_real": 91.01,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "r-s",
        "precision_real": 9.76,
        "recall_real": 100.0,
        "f1score_real": 17.78,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "r-m",
        "precision_real": 28.24,
        "recall_real": 100.0,
        "f1score_real": 44.04,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "r-l",
        "precision_real": 32.56,
        "recall_real": 100.0,
        "f1score_real": 49.12,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "rt-m",
        "precision_real": 22.22,
        "recall_real": 100.0,
        "f1score_real": 36.36,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "rt-l",
        "precision_real": 93.94,
        "recall_real": 100.0,
        "f1score_real": 96.88,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 91.3,
        "f1score_real": 95.45,
        "time": ""
    },
    {
        "model": "DTL",
        "train": "rtd",
        "test": "rtd-l",
        "precision_real": 94.19,
        "recall_real": 100.0,
        "f1score_real": 97.01,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "r-m",
        "precision_real": 56.0,
        "recall_real": 63.64,
        "f1score_real": 59.57,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "r-l",
        "precision_real": 30.23,
        "recall_real": 100.0,
        "f1score_real": 46.43,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "rt-m",
        "precision_real": 35.0,
        "recall_real": 41.18,
        "f1score_real": 37.84,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "rt-l",
        "precision_real": 80.0,
        "recall_real": 23.81,
        "f1score_real": 36.7,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "rtd-m",
        "precision_real": 41.18,
        "recall_real": 38.89,
        "f1score_real": 40.0,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "r",
        "test": "rtd-l",
        "precision_real": 90.0,
        "recall_real": 88.73,
        "f1score_real": 89.36,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "r-m",
        "precision_real": 43.33,
        "recall_real": 59.09,
        "f1score_real": 50.0,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "r-l",
        "precision_real": 28.89,
        "recall_real": 100.0,
        "f1score_real": 44.83,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "rt-m",
        "precision_real": 84.21,
        "recall_real": 94.12,
        "f1score_real": 88.89,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "rt-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "rtd-m",
        "precision_real": 30.56,
        "recall_real": 61.11,
        "f1score_real": 40.74,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rt",
        "test": "rtd-l",
        "precision_real": 78.89,
        "recall_real": 100.0,
        "f1score_real": 88.2,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "r-m",
        "precision_real": 38.71,
        "recall_real": 54.55,
        "f1score_real": 45.28,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "r-l",
        "precision_real": 28.09,
        "recall_real": 96.15,
        "f1score_real": 43.48,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "rt-m",
        "precision_real": 36.84,
        "recall_real": 82.35,
        "f1score_real": 50.91,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "rt-l",
        "precision_real": 93.26,
        "recall_real": 98.81,
        "f1score_real": 95.95,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "rtd-m",
        "precision_real": 84.21,
        "recall_real": 88.89,
        "f1score_real": 86.49,
        "time": ""
    },
    {
        "model": "CNN",
        "train": "rtd",
        "test": "rtd-l",
        "precision_real": 98.61,
        "recall_real": 100.0,
        "f1score_real": 99.3,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "r-m",
        "precision_real": 100.0,
        "recall_real": 95.0,
        "f1score_real": 97.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "rt-l",
        "precision_real": 100.0,
        "recall_real": 96.0,
        "f1score_real": 98.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "r",
        "test": "rtd-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "r-m",
        "precision_real": 92.0,
        "recall_real": 82.0,
        "f1score_real": 85.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "rt-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rt",
        "test": "rtd-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "r-m",
        "precision_real": 97.0,
        "recall_real": 95.0,
        "f1score_real": 96.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "rt-l",
        "precision_real": 100.0,
        "recall_real": 94.0,
        "f1score_real": 97.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "Classify",
        "train": "rtd",
        "test": "rtd-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "r-s",
        "precision_real": 0.0,
        "recall_real": 0.0,
        "f1score_real": 0.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "r-m",
        "precision_real": 100.0,
        "recall_real": 77.27,
        "f1score_real": 87.18,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rt-m",
        "precision_real": 66.67,
        "recall_real": 87.5,
        "f1score_real": 75.68,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rt-l",
        "precision_real": 94.38,
        "recall_real": 100.0,
        "f1score_real": 97.11,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rtd-m",
        "precision_real": 73.91,
        "recall_real": 80.95,
        "f1score_real": 77.27,
        "time": ""
    },
    {
        "model": "PU",
        "train": "r",
        "test": "rtd-l",
        "precision_real": 96.05,
        "recall_real": 100.0,
        "f1score_real": 97.99,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "r-s",
        "precision_real": 0.0,
        "recall_real": 0.0,
        "f1score_real": 0.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "r-m",
        "precision_real": 100.0,
        "recall_real": 81.82,
        "f1score_real": 90.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rt-m",
        "precision_real": 100.0,
        "recall_real": 87.5,
        "f1score_real": 93.33,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rt-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 90.48,
        "f1score_real": 95.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rt",
        "test": "rtd-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "r-s",
        "precision_real": 0.0,
        "recall_real": 0.0,
        "f1score_real": 0.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "r-m",
        "precision_real": 100.0,
        "recall_real": 72.73,
        "f1score_real": 84.21,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "r-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rt-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rt-m",
        "precision_real": 71.43,
        "recall_real": 93.75,
        "f1score_real": 81.08,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rt-l",
        "precision_real": 95.45,
        "recall_real": 100.0,
        "f1score_real": 97.67,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rtd-s",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rtd-m",
        "precision_real": 100.0,
        "recall_real": 95.24,
        "f1score_real": 97.56,
        "time": ""
    },
    {
        "model": "PU",
        "train": "rtd",
        "test": "rtd-l",
        "precision_real": 100.0,
        "recall_real": 100.0,
        "f1score_real": 100.0,
        "time": ""
    }
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
    formatPercentage(value) {
    if (typeof value !== 'number') {
      return '';
    }
    return value.toFixed(2) + '%';
  },
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
          this.testTime = (endTime - startTime) / 1000; // 转换为秒
          console.log("testtime", this.testTime);
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
      this.initTableData();
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
