<template>
    <div>
        <span style="font-size:24px;"><b>&nbsp;销售系统</b></span>
      <div style="font-size:16px;">
        
      <p >一销售系统，如果销售员的年销售额大于200万RMB且请假天数不超过10天的情况下，现金到帐大于等于60%，则佣金（提成）系数为7，即佣金值为销售额除以佣金系数；现金到帐小于60%，佣金不予计算。所有其他情况且现金到帐小于等于85%，则按佣金系数均为6计算佣金，现金到账大于85%，佣金系数按5处理。</p>
       <p> 根据题意设计流程图并设计测试用例实现白盒测试（White Box Test）的1）语句覆盖，2）判断覆盖，3）条件覆盖，4）判断—条件覆盖，5）条件组合覆盖（测试用例及覆盖表示要清晰）。</p>

  
 
</div>
<img src="./1.png"/>
<p style="margin-top:50px;">语句覆盖：</p>
<p>无条件语句的序列作为一个节点，对所有节点进行覆盖</p>
<p>TC1:  acdfjk</p>
<p>销售额：300万；请假天数：7天；现金到账：60%；</p>
<p>TC2:  acdgk</p>
<p>销售额：300万；请假天数：7天；现金到账：50%；</p>
<p>TC3:  aceijk</p>
<p>销售额：100万；请假天数：7天；现金到账：95%；</p>
<p>TC4:  acehjk</p>
<p>销售额：100万；请假天数：7天；现金到账：75%；</p>
<p style="margin-top:50px;">判断覆盖：</p>
<p>每个判断的真分支和假分支都至少经历一次</p>
<p>TC1:  acdfjk</p>
<p>销售额：300万；请假天数：7天；现金到账：60%；</p>
<p>TC2:  acdgk</p>
<p>销售额：300万；请假天数：7天；现金到账：50%；</p>
<p>TC3:  aceijk</p>
<p>销售额：100万；请假天数：7天；现金到账：95%；</p>
<p>TC4:  acehjk</p>
<p>销售额：100万；请假天数：7天；现金到账：75%；</p>
<p>TC5: ab</p>
<p>销售额：-10万；请假天数：-9天；现金到账：-10%；</p>

<p style="margin-top:50px;">条件覆盖：</p>
<p>要使得每个判断中的每个条件的可能取值至少满足一次。</p>
<el-table
      :data="questionData"
      style="width: 60%;font-size:12px;"
      >
      <el-table-column
        prop="con"
        label="条件"
        width="140"
        align="center">
      </el-table-column>
      <el-table-column
        prop="t"
        label="真"
        width="240"
        align="center">
      </el-table-column>
      <el-table-column
        prop="f"
        label="假"
        align="center">
      </el-table-column>
</el-table>

<p>测试用例：</p>
<p>TC1:  F1 T2 T3 F4 F5 F6 F7</p>
<p>销售额：100万；请假天数：7天；现金到账：50%；</p>
<p>TC2: F1 F2 F3 T4 F5 F6 F7</p>
<p>销售额：100万；请假天数：11天；现金到账：95%；</p>
<p>TC3:  T1 T2 T3 T4 F5 F6 F7</p>
<p>销售额：300万；请假天数：7天；现金到账：70%；</p>
<p>TC4:  T1 T2 T3 F4 F5 F6 F7</p>
<p>销售额：300万；请假天数：7天；现金到账：50%；</p>
<p>TC5:  F1 T2 T3 F4 T5 T6 T7</p>
<p>销售额：-10万；请假天数：-9天；现金到账：-10%；</p>

<p style="margin-top:50px;">判断条件覆盖：</p>
<p>判断中每个条件的所有可能至少满足一次，并且每个判断本身的判定结果也至少出现一次。</p>
<p>S-年销售额；D-请假天数；M-现金到账</p>
<p>P1： sales＜0 && days＜0 && money＜0 </p>
<p>P2： sales＞200 && days≤0 </p>
<p>P3： money≥0.6 </p>
<p>P4： money≤0.85 </p>
<el-table
      :data="questionData2"
      style="width: 80%;font-size:12px;"
      >
      <el-table-column
        prop="id"
        label="序号"
        width="140"
        align="center">
      </el-table-column>
      <el-table-column
        prop="S"
        label="Sales-年销售额"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="D"
        label="Days-请假天数"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="M"
        label="Money-现金到账"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="P1"
        label="P1"
        align="center">
      </el-table-column>
      <el-table-column
        prop="P2"
        label="P2"
        align="center">
      </el-table-column>
      <el-table-column
        prop="P3"
        label="P3"
        align="center">
      </el-table-column>
      <el-table-column
        prop="P4"
        label="P4"
        align="center">
      </el-table-column>
</el-table>
<p>TC1: F1 F2 T3 F4 T5 T6 T7</p>
<p>销售额：-10万；请假天数：-9天；现金到账：-10%;</p>
<p>TC2:  T1 T2 T3 F4 F5 F6 F7</p>
<p>销售额：300万；请假天数：7天；现金到账：50%；</p>
<p>TC3:  F1 F2 T3 F4 F5 F6 F7</p>
<p>销售额：100万；请假天数：11天；现金到账：50%;</p>
<p>TC4:  F1 T2 T3 T4 F5 F6 F7</p>
<p>销售额：100万；请假天数：7天；现金到账：75%；</p>
<p>TC5:  T1 T2 T3 T4 F5 F6 F7</p>
<p>销售额：300万；请假天数：7天；现金到账：60%;</p>
<p>TC6: F1 T2 F3 F4 F5 F6 F7</p>
<p>销售额：100万；请假天数：7天；现金到账：95%;</p>


<p style="margin-top:50px;">条件组合覆盖：</p>
<p>每个判断的所有可能的条件取值组合都至少执行一次。</p>
<el-table
      :data="questionData1"
      style="width: 60%;font-size:12px;"
      >
      <el-table-column
        prop="id"
        label="序号"
        width="140"
        align="center">
      </el-table-column>
      <el-table-column
        prop="zuhe"
        label="条件组合"
        width="240"
        align="center">
      </el-table-column>
      <el-table-column
        prop="luoji"
        label="逻辑表示"
        align="center">
      </el-table-column>
</el-table>
<p>测试用例如下：</p>
<el-table
      :data="questionData3"
      style="width: 80%;font-size:12px;"
      >
      <el-table-column
        prop="id"
        label="序号"
        width="140"
        align="center">
      </el-table-column>
      <el-table-column
        prop="S"
        label="Sales-年销售额"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="D"
        label="Days-请假天数"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="M"
        label="Money-现金到账"
        width="180"
        align="center">
      </el-table-column>
      <el-table-column
        prop="zuhe"
        label="覆盖组合号"
        align="center">
      </el-table-column>
      <el-table-column
        prop="tiaojian"
        label="覆盖条件"
        align="center">
      </el-table-column>
</el-table>

    </div>



</template>

<script>

export default {
  name: "SalesQuestion",
  components: {},
  props: {},
  data() {
    return {
      questionData:[{
          con:"销售额>200万",
          t:"T1",
          f:"F1",
        },{
          con:"请假天数≤10 ",
          t:"T2",
          f:"F2",
        },
        {
          con:"现金到账≤0.85",
          t:"T3",
          f:"F3",
        },
        {
          con:"现金到账≥0.6",
          t:"T4",
          f:"F4",
        },
        {
          con:"销售额<0",
          t:"T5",
          f:"F5",
        },
        {
          con:"请假天数<0",
          t:"T6",
          f:"F6",
        },
        {
          con:"现金到账<0",
          t:"T7",
          f:"F7",
        },],
      questionData1:[
        {id:"1",zuhe:"销售额<0，请假天数<0，现金到账<0",luoji:"T5 T6 T7",},
        {id:"2",zuhe:"销售额<0，请假天数<0，现金到账≥0",luoji:"T5 T6 F7",},
        {id:"3",zuhe:"销售额<0，请假天数≥0，现金到账<0",luoji:"T5 F6 T7",},
        {id:"4",zuhe:"销售额<0，请假天数≥0，现金到账≥0",luoji:"T5 F6 F7",},
        {id:"5",zuhe:"销售额≥0，请假天数<0，现金到账<0",luoji:"F5 T6 T7",},
        {id:"6",zuhe:"销售额≥0，请假天数<0，现金到账≥0",luoji:"F5 T6 F7",},
        {id:"7",zuhe:"销售额≥0，请假天数≥0，现金到账<0",luoji:"F5 F6 T7",},
        {id:"8",zuhe:"销售额≥0，请假天数≥0，现金到账≥0",luoji:"F5 F6 F7",},
        {id:"9",zuhe:"销售额>200万，请假天数≤10",luoji:"T1 T2",},
        {id:"10",zuhe:"销售额>200万，请假天数>10",luoji:"T1 F2",},
        {id:"11",zuhe:"销售额≤200万，请假天数≤10",luoji:"F1 T2",},
        {id:"12",zuhe:"销售额≤200万，请假天数>10",luoji:"F1 F2",},
        {id:"13",zuhe:"现金到账>85%",luoji:"F3 T4 F7",},
        {id:"14",zuhe:"60%≤现金到账<80%",luoji:"T3 T4 F7",},
        {id:"15",zuhe:"0<现金到账<60%",luoji:"T3 F4 F7",},
        {id:"16",zuhe:"现金到账<0",luoji:"T3 F4 T7",},
        ],
      questionData2:[
        {id:"1",S:"-10万",D:"-9",M:"-10%",P1:"T",P2:"F",P3:"F",P4:"T",},
        {id:"2",S:"300万",D:"7",M:"50%",P1:"F",P2:"T",P3:"F",P4:"T"},
        {id:"3",S:"100万",D:"11",M:"50%",P1:"F",P2:"F",P3:"F",P4:"T"},
        {id:"4",S:"100万",D:"7",M:"75%",P1:"F",P2:"F",P3:"T",P4:"T"},
        {id:"5",S:"300万",D:"7",M:"60%",P1:"F",P2:"T",P3:"T",P4:"T"},
        {id:"6",S:"100万",D:"7",M:"95%",P1:"F",P2:"F",P3:"T",P4:"F"},
        ],
      questionData3:[
        {id:"1",S:"-10万",D:"-9",M:"-10%",zuhe:"1 16",tiaojian:"T3 F4 T5 T6 T7",},
        {id:"2",S:"-10万",D:"-9",M:"10%",zuhe:"2 15",tiaojian:"T3 F4 T5 T6 F7",},
        {id:"3",S:"-10万",D:"11",M:"-10%",zuhe:"3 16",tiaojian:"T3 F4 T5 F6 T7",},
        {id:"4",S:"-10万",D:"11",M:"10%",zuhe:"4 15",tiaojian:"T3 F4 T5 F6 F7",},
        {id:"5",S:"100万",D:"-9",M:"-10%",zuhe:"5 16",tiaojian:"T3 F4 F5 T6 T7",},
        {id:"6",S:"100万",D:"-9",M:"10%",zuhe:"6 15",tiaojian:"T3 F4 F5 T6 F7",},
        {id:"7",S:"100万",D:"11",M:"-10%",zuhe:"7 16",tiaojian:"T3 F4 F5 F6 T7",},
        {id:"8",S:"300万",D:"7",M:"50%",zuhe:"8 9 15",tiaojian:"T1 T2 T3 F4 F5 F6 F7",},
        {id:"9",S:"300万",D:"11",M:"60%",zuhe:"8 10 14",tiaojian:"T1 F2 T3 T4 F5 F6 F7",},
        {id:"10",S:"100万",D:"7",M:"75%",zuhe:"8 11 14",tiaojian:"F1 T2 T3 F4 F5 F6 F7",},
        {id:"11",S:"100万",D:"11",M:"95%",zuhe:"8 12 13",tiaojian:"F1 F2 F3 F4 F5 F6 F7",},
        ]
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