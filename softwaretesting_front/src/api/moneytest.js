import request from "../utils/request"

export const testmoney = data => {
  return request({
    method: 'POST',
    url: '/api/money/test',
    data
  })
}

export const uploadfile = data => {
  return request({
    method: 'POST',
    config: { 'Content-Type': 'multipart/form-data' },
    data,
    responseType: 'blob',
    url: '/api/money/upload',
  })
}