<template>
    <div>
        <span style="font-size:24px;"><b>&nbsp;电商平台测试</b></span>
      <div style="font-size:16px;">
        
      <p >讨论题：边界值测试可以适用于function(method) level，class level,，system level，分析电商平台系统测试时考虑的边界值情况。</p>

</div>

  <p style="margin-top:100px;font-size:20px;">一. 问题分析</p>
  <p>电商平台流程图</p>
  <img src="./5.png"/>
  <p style="margin-top:50px;">二. Method level</p>
<p>输入/输出变量取值边界，循环次数，选项边界值等</p>
  <p style="margin-top:50px;">三. Class level</p>
<p>考虑类内方法，类与类之间接口调用的变量边界等</p>

  <p style="margin-top:100px">四. System level</p>
  <p >1.功能测试</p>
  <p>测试依据为需求文档及需求分析文档，验证产品功能是否符合需求。</p>
<el-table
      :data="questionData"
      style="width: 60%;font-size:12px;"
      >
      <el-table-column
        prop="minute"
        label=""
        width="140"
        align="center">
      </el-table-column>
      <el-table-column
        prop="times"
        label="正常情况"
        width="240"
        align="center">
      </el-table-column>
      <el-table-column
        prop="level"
        label="边界值"
        align="center">
      </el-table-column>
    </el-table>
<p style="margin-top:50px;">举例：</p>
<p>用户密码输入次数限制为5：那么我们根据基本边界值需要测试1，2，3，4，5如果考虑健壮性还需要测试0和6</p>
<p style="margin-top:50px;">2.非功能需求</p>
<p>兼容性测试：考虑不同浏览器版本和不同操作系统版本之间的边界值</p>
<p>性能测试：按照需求规约所提出的性能要求，考虑吞吐率和响应时间边界值情况</p>
<p>压力测试：考虑大型电商活动如618等活动期间，是否满足系统预先要求的极限</p>



    </div>



</template>

<script>

export default {
  name: "SalesQuestion",
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