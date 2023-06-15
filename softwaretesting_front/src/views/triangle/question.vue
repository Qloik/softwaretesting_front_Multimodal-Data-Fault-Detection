<template>
  <div>
    <span style="font-size: 24px"><b>&nbsp;判断三角形问题</b></span>
    <div style="font-size: 16px">
      <b><p>问题描述</p></b>
      <p>
        输入三个整数a、b、c，分别作为三角形的三条边，判断这三条边是否能构成三角形；如果能构成三角形，则判断三角形的类型（等边三角形、等腰三角形、直角三角形、一般三角形）。
      </p>
      <p>
        注：为了便于讨论，假设a、b、c的范围为0.01&lt;=a&lt;=800, 0.01&lt;=b&lt;=800, 0.01&lt;=c&lt;=800，这与中文教材上略有不同，可以取到边界值
      </p>
      <b><p>问题分析</p></b>
      <p>边界值分析法：采用健壮性边界值分析，共有6n+1=19个测试用例。</p>
      <p>等价类法：采用强健壮等价类设计测试用例，弱一般与强一般对应的测试用例均为4个，额外强健壮等价类有C(3,1)C(2,1)+C(3,2)C(2,1)C(2,1)
        +C(3,3)C(2,1)C(2,1)C(2,1)有26个测试用例
      </p>
    </div>
  </div>
</template>

<script>

export default {
  name: "TriangleQuestion",
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