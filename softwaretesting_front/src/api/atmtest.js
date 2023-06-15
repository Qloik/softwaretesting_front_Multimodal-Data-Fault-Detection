import request from "../utils/request"

export const testatm = data => {
  return request({
    method: 'POST',
    url: '/api/atm/test',
    data
  })
}

export const uploadfile = data => {
  return request({
    method: 'POST',
    config: { 'Content-Type': 'multipart/form-data' },
    data,
    responseType: 'blob',
    url: '/api/atm/upload',
  })
}