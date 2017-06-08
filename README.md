# 使用SVR进行多时间尺度商业负荷预测
  
我使用了历史用电数据、天气数据（温度和湿度，数据从[Wunderground.com](https://www.wunderground.com/)获得）、一天中的时刻（0~23）、日期类型（是否节假日）
作为支持向量机的输入特征向量。预测时间尺度有15min、30min、1h、2h。SVR的算法采用python的库[sklearn](http://scikit-learn.org/)。特别感谢用户[@jtelszasz](https://github.com/jtelszasz/my_energy_data_viz)的工作，给了我很大启发。
  
- elec_data  
包含了用电数据，出于商业机密考虑，未上传至此。
- weather_data  
包含了从[Wunderground](https://www.wunderground.com/)爬的天气数据。里面有一个json格式的数据文件，可用来预览该网站返回的数据格式。  
- my_functions  
  - scrape_weather_data: 从[Wunderground](https://www.wunderground.com/)爬的天气数据的脚本。
  - read_data: 将用电数据和电气输入读入pandas.DataFrame，进行不同时间尺度的聚合
  - my_errors：误差计算的几个函数，包括MAE, MAPE, RMSE
  - my_svr：数据标准化，SVR训练，预测，网格搜索
  - data_visualization：画图的一些函数
- hac.ipynb | hhy.ipynb  
某商场和某酒店的预测分析。