<template>
    <div>
        <span style="font-size:24px;"><b>&nbsp;项目测试</b></span>
      <div style="font-size:16px;">
        
      <p >找一个项目（或软件工程课程项目）部署一下，分析边界值并设计测试用例执行之。</p>

    </div>

  <h3 style="margin-top:100px">一. 问题分析</h3>
  <p>项目用例图</p>
  <img src="./usecase_1.png" style="width:100%"/>
  <h3 style="margin-top:50px;">二. 边界值分析</h3>
  <p>对于完成一场约球获取积分的数量，我们会根据不同输入的边界值进行判断，包括参与的场次，是否是公开约球，信誉度等进行不同的输入测试</p>
<h3 style="margin-top:50px;">三. 测试用例</h3>
<img src="./test.jpg" style="width:100%">

    </div>



</template>

<script>

export default {
  name: "project",
  components: {},
  props: {},
  data() {
    return {questionData:[{
          minute:"登录",
          times:"登录成功",
          level:"账号/密码有误导致重新输入次数超过限制;自动注销时间限制",
        },{
          minute:"搜索商品",
          times:"显示符合条件的商品",
          level:"输入信息没有找到，结果为空",
        },{
          minute:"购买商品使用优惠券",
          times:"使用成功",
          level:"不符合要求；优惠券张数限制",
        },{
          minute:"加入购物车",
          times:"加入成功",
          level:"购物车商品数量限制",
        },{
          minute:"添加地址",
          times:"添加成功",
          level:"地址无效",
        },{
          minute:"取消订单",
          times:"取消成功",
          level:"取消时间限制;退款金额限制",
        },
        {
          minute:"物流",
          times:"确认收货",
          level:"自动确认时间限制;商品丢失;商品损坏;超过预计时间;退货/换货",
        },
       
       
        ], }
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