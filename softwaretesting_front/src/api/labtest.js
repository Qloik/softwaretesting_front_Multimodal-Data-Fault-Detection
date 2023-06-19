import request from "../utils/request"

export const testlab = data => {
  return request({
    method: 'POST',
    url: '/process',
    data
  })
}

export const uploadfile = data => {
  return request({
    method: 'POST',
    config: { 'Content-Type': 'multipart/form-data' },
    data,
    responseType: 'blob',
    url: '/api/lab/upload',
  })
}

// {"id":"TS5","model":2099,"train_data":6,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS6","model":2100,"train_data":6,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS7","model":2101,"train_data":6,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS8","model":2022,"train_data":0,"data":0,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS9","model":2022,"train_data":1,"data":1,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS10","model":2022,"train_data":2, "data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS11","model":2022,"train_data":11,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS12","model":2022,"train_data":12,"data":0,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS13","model":2022,"train_data":13,"data":1,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS14","model":2022,"train_data":6,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS15","model":2022,"train_data":6,"data":6,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS16","model":2022,"train_data":6,"data":0,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS17","model":2022,"train_data":6,"data":1,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS18","model":2022,"train_data":6,"data":31,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"},
// {"id":"TS19","model":2022,"train_data":6,"data":32,"expectation":"-","precision":"0.9997","recall":"0.9745","f1-score":"0.9870","info":"","state":"测试通过","time":"1.512s"}
