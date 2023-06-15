<template>
  <div class="file">
    <el-button type="primary" plain @click="outExe">生成模板</el-button>

    <el-upload
      class="upload-demo"
      drag
      action="https://jsonplaceholder.typicode.com/posts/"
      multiple
      :file-list="fileList"
      :http-request="getFile"
    >
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      <div class="el-upload__tip" slot="tip">
        请先点击生成模板，填写好相应测试用例，再上传，限xls/xlsx格式，不超过30M.
      </div>
    </el-upload>
  </div>
</template>

<script>
import printExe from "@/excel/outexe.js";
import { uploadfile } from "@/api/moneytest.js";
import { dateformat } from "@/utils/dateformat.js";
export default {
  name: "ExcelFile",
  components: {},
  props: {},
  data() {
    return {
      fileList: [],
    };
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {
    outExe() {
      this.$confirm("此操作将导出excel文件, 是否继续?", "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      })
        .then(() => {
          let tHeader = [
            "测试用例编号",
            "年销售额",
            "请假天数",
            "现金到账",
            "佣金系数预期输出",
            "佣金系数实际输出",
            "程序运行信息",
            "测试结果",
            "测试时间",
          ];

          const filterVal = [
            "id",
            "sales",
            "days",
            "money",
            "expectation",
            "actual",
            "info",
            "test_result",
            "test_time",
          ];

          const example = [
            {
              id: "TS1",
              sales: "300",
              days: "7",
              money: "60",
              expectation: "7",
            },
          ];
          printExe("销售系统问题模板", tHeader, filterVal, example);
        })
        .catch(() => {
          this.$message("已取消");
        });
    },
    getFile(item) {
      let formData = new FormData();
      formData.append("file", item.file);
      uploadfile(formData)
        .then((res) => {
          let url = window.URL.createObjectURL(new Blob([res.data]));
          let a = document.createElement("a");
          a.style.display = "none";
          a.href = url;
          a.setAttribute(
            "download",
            "销售系统问题测试报告 " + dateformat() + ".xls"
          );
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
        })
        .catch(() => {
          this.$message.error("Server Error");
        });
    },
  },
};
</script>

<style scoped>
.upload-demo {
  width: 80%;
  margin-top: 20px;
}
</style>
