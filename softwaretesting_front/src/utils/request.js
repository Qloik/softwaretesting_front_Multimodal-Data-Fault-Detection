

import axios from 'axios';
// import MockAdapter from 'axios-mock-adapter';

// const mock = new MockAdapter(axios);

// // 模拟数据
// const mockData = {
//   test_result: [
//     {
//       precision_real: 0.85,
//       recall_real: 0.9,
//       f1score_real: 0.87,
//       test_result: true,
//       test_time: 1.402
//     },
//     {
//         precision_real: 0.85,
//         recall_real: 0.9,
//         f1score_real: 0.87,
//         test_result: true,
//         test_time: 1.402
//       },
//       {
//         precision_real: 0.85,
//         recall_real: 0.9,
//         f1score_real: 0.87,
//         test_result: true,
//         test_time: 1.402
//       },
//       {
//         precision_real: 0.85,
//         recall_real: 0.9,
//         f1score_real: 0.87,
//         test_result: true,
//         test_time: 1.402
//       },
//       {
//         precision_real: 0.85,
//         recall_real: 0.9,
//         f1score_real: 0.87,
//         test_result: true,
//         test_time: 1.402
//       },

//        {
//       precision_real: 0.85,
//       recall_real: 0.9,
//       f1score_real: 0.87,
//       test_result: true,
//       test_time: 1.402
//     },
//   ]
// };

// // 拦截 /api/lab/test 请求，并返回模拟数据
// mock.onPost('/api/lab/test').reply(200, mockData);



// 实验后端
export const request = axios.create({

    baseURL:'http://localhost:5000/'
    
  // baseURL: 'http://localhost:5001'

});

// 项目后端
export const request1 = axios.create({

  baseURL:'http://localhost:5001/'
  
// baseURL: 'http://localhost:5000'

});

export default request;