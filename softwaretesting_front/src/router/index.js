import Vue from 'vue'
import VueRouter from 'vue-router'
import Login from '@/views/login/'
import Home from '@/views/home/'
import Layout from '@/views/layout/'
import Cash from '@/views/cash/'
import Triangle from '@/views/triangle'
import Sales from '@/views/sales/'


import three from '@/views/three/'
import five from '@/views/five/'
import eight from '@/views/eight/'
import Nine from '@/views/nine/'
import thirteen from '@/views/thirteen/'
import fourteen from '@/views/fourteen/'
import sixteen from '@/views/sixteen/'
import sale from '@/views/sale/'
import Atm from '@/views/atm/'
import six from "@/views/six"
Vue.use(VueRouter)

const routes = [{
    path: '/login',
    name: 'login',
    component: Login
  },
  {
    path: '/',
    component: Layout,
    children: [{
        path: '',
        name: 'home',
        component: Home
      },
      {
        path: '/cash',
        name: 'cash',
        component: Cash
      }, {
        path: '/triangle',
        name: 'triangle',
        component: Triangle
      },
      {
        path: '/calendar',
        name: 'calendar',
        component: () => import("@/views/calendar")
      },
      {
        path: '/sales',
        name: 'sales',
        component: Sales
      },
      {
        path: '/three',
        name: 'three',
        component: three
      },
      {
        path: '/five',
        name: 'five',
        component: five
      },
      {
        path: '/eight',
        name: 'eight',
        component: eight
      },
      {
        path: '/nine',
        name: 'nine',
        component: Nine
      },
      {
        path: '/thirteen',
        name: 'thirteen',
        component: thirteen
      },
      {
        path: '/fourteen',
        name: 'fourteen',
        component: fourteen
      },
      {
        path: '/sixteen',
        name: 'sixteen',
        component: sixteen
      },
      {
        path: '/sale',
        name: 'sale',
        component: sale
      },
      {
        path: '/atm',
        name: 'atm',
        component: Atm
      },
      {
        path: '/six',
        name: 'six',
        component: six
      },
    ]
  }
]

const router = new VueRouter({
  routes
})

const user = JSON.parse(window.localStorage.getItem('user'));
//导航守卫
// router.beforeEach((to,_,next) =>{
//   //校验非登录页面的登录状态
//   if(to.path !== '/login'){
//     if(user){
//       next();
//     }else{
//       next('./login');
//     }
//   }else{
//     //登录页面正常允许通过
//     next()
//   }
// })

export default router