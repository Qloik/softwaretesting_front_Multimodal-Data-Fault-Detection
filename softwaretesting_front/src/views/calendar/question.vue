<template>
  <div>
    <span style="font-size: 24px"><b>&nbsp;万年历问题</b></span>
    <div style="font-size: 17px">
      <p>在本题中，我们需要根据输入的年月日三个正整数变量，
        若该日期是一个合法日期，则以"yyyy-mm-dd"的形式输出该日期的下一天，
        若该日期不是一个合法日期，则应输出-1。
      </p>
    </div>
    <span style="font-size: 24px"><b>&nbsp;—————————</b></span>
    <div style="font-size: 17px">
      <p></p>
    </div>
    <span style="font-size: 24px"><b>&nbsp;决策法说明</b></span>
    <div style="font-size: 16px">
      <p><b>&nbsp;等价类</b></p>
      <p> Y1 = 非闰年2022 </p>
      <p> Y2 = 被4整除不被100整除的闰年2004 </p>  
      <p> Y3 = 被400整除的闰年2000 </p>  
      <p> M1 = 2月 </p>  
      <p> M2 = 12月 </p>  
      <p> M3 = 1、3、5、7、8、10月 </p>  
      <p> M4 = 4、6、9、11月 </p>  
      <p> D1 = 28 </p>  
      <p> D2 = 29 </p>  
      <p> D3 = 30 </p>  
      <p> D4 = 31 </p> 
      <p> D5 = 其他日 </p>  
      <b>&nbsp;1.条件桩</b>
      <p> C1 年在{Y1 Y2 Y3}中 </p>
      <p> C2 月在{M1M2 M3 M4}中 </p>
      <p> C3 日在{D1 D2 D3 D4 D5}中 </p>
      <b>&nbsp;2.动作桩</b>
      <p> D1 下一日期 </p>
      <b>&nbsp;3.确定规则的个数</b>
      <p> 3*4*5=60 </p>
      <b>&nbsp;4.填入条件项和动作项，得到初始决策表</b>
    </div>
    <img src="./1.png" style="width:100%;height:100%;
    object-fit: fill;object-fit: contain;object-fit: scale-down;" />

    <div style="font-size: 16px">
      <p>
          <b>&nbsp;5.简化决策表，合并相似规则</b>
      </p>
    </div>
    <img src="./2.png" style="width:100%;height:100%;
    object-fit: fill;object-fit: contain;object-fit: scale-down;" />
  </div>
</template>

<script>
export default {
  name: "CalendarQuestion",
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