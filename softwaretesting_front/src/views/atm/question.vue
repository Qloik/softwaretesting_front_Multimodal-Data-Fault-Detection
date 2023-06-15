<template>
  <div style="overflow-y: scroll">
    <span style="font-size: 24px"><b>&nbsp;ATM问题</b></span>
    <div style="font-size: 17px">
      <p>分析或查阅得到正确的ATM的系统状态图，再基于state transition testing分析出transition tree，最后设计出逻辑测试用例。
      </p>
      <p style="margin-top:100px;font-size:20px;">状态图：</p>
      <img style="width:60%;height:60%;" src="./ATM状态图.png"/>
      <p style="margin-top:100px;font-size:20px;">Transition Tree：</p>
      <img style="width:60%;height:60%;" src="./atm-transition-tree.png"/>
    </div>
  </div>
</template>

<script>
export default {
  name: "AtmQuestion",
  components: {},
  props: {},
  data() {
    return {
    };
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {},
};
</script>

<style scoped lang="less">
</style>