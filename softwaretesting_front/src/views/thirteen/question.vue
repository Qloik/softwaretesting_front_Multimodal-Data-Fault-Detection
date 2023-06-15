<template>
  <div>
    <span style="font-size:24px;"><b>&nbsp;ERP系统</b></span>
    <div style="font-size:16px;">
      
    <p >下图是ERP系统的流图，根据系统需求和流程图，归纳出比较清晰的主、备选流关系图，并基于主、备选流设计场景和相应的测试用例。 MPS（ Master Production Schedule），MRP(Manufacturing Resource Planning)</p>
    <img style="width:30%;height:30%;" src="./题目.png"/>
  </div>
  <p>程序的基本流：</p>
  <p>根据销售合同制定计划需求；进行MPS计算；此时可用库存不满足销售需求，且由于没有可用库存，直接进入主生产计划；进行MRP计算，发现可用库存不满足生产需求；计划请购单；计划采购订单；采购订单；采购到货处理；采购质检，质检合格；按计划生产订单；制定车间作业计划；进行生产工序管理，工序转移，完工处理；进行生产质检，质检合格；将生产品入库；进行销售发货管理；产品发货直至发完；合同结案</p>
  <p>备选流：</p>
  <p>1. 可用库存满足销售需求</p>
  <p>2. 有可用库存</p>
  <p>3. 可用库存满足生产需求</p>
  <p>4. 采购质检不合格</p>
  <p>5. 生产质检不合格</p>
  <p>6. 发货未完成</p>
  <p>示意图如下</p>
  <img style="width:60%;height:60%;" src="./示意图.png"/>
  <p>根据基本流和备选流生成如下场景:</p>
  <img style="width:60%;height:60%;" src="./scene.png"/>
  <p>对应测试用例如下:</p>
  <img style="width:60%;height:60%;" src="./test-case.png"/>
</div>



</template>

<script>

export default {
  name: "SalesQuestion",
  components: {},
  props: {},
  data() {
    return {
    }
  },
  computed: {},
  watch: {},
  created() {},
  mounted() {},
  methods: {

  },
};
</script>

<style scoped lang="less">


</style>