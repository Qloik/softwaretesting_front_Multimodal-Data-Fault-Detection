<template>
  <div class="single-case">
    <el-form
      class="single-form"
      :label-position="labelPosition"
      label-width="50px"
      :model="formLabelAlign"
    >
      <el-form-item label="Model">
        <div class="button-row">
          <el-radio-group v-model="formLabelAlign.model">
            <el-radio label="1" size="large" border>Model A</el-radio>
            <el-radio label="2" size="large" border>Model B</el-radio>
            <el-radio label="3" size="large" border>Model C</el-radio>
            <el-radio label="4" size="large" border>Model D</el-radio>
          </el-radio-group>
        </div>
        <!-- <el-input v-model=""></el-input> -->
      </el-form-item>

      <el-form-item label="Train">
        <!-- <el-input v-model="formLabelAlign.train"></el-input> -->
        <el-select
          v-model="formLabelAlign.train"
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
      </el-form-item>

      <el-form-item label="Test">
        <!-- <el-input v-model="formLabelAlign.test"></el-input> -->
        <el-select
          v-model="formLabelAlign.test"
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
      <span>实际输出：{{ actual }}</span>
      <el-divider direction="vertical"></el-divider>
      <span>运行信息：{{ info }}</span>
      <el-divider direction="vertical"></el-divider>
    </div>
  </div>
</template>

<script>
import { testcalendar } from "@/api/calendartest.js";
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
      formLabelAlign: {
        model: ref("1"),
        train: ref(""),
        test: ref(""),
      },
      date: "",
      loading: false,
      options1: [
        {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
        },
      ],
      options2: [
        {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
        },
         {
          value: "Option1",
          label: "Option1",
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
      let formdata = {
        id: "TS1",
        year: this.formLabelAlign.year,
        month: this.formLabelAlign.month,
        day: this.formLabelAlign.day,
        expectation: this.formLabelAlign.expectation,
      };
      let data = {
        calendar_test_list: [formdata],
      };
      testcalendar(data).then((res) => {
        this.actual = res.data.test_result[0].actual;
        this.info = res.data.test_result[0].info;
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

.button-row {
  display: flex;
}

.el-radio-group {
  display: flex;
  align-items: center;
}
</style>
