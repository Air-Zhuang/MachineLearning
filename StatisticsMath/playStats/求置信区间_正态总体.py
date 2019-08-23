from math import sqrt
from scipy.stats import norm,t,chi2
from playStats.descriptive_stats import mean,std,variance

def mean_ci_est(data,alpha,sigma=None):
    """
        总体方差未知，求均值的置信空间
        总体方差已知，求均值的置信空间
        data为传入的样本; alpha,sigma为需要传入的置信水平,sigma的值
    """
    n=len(data)                                     #求样本容量
    sample_mean=mean(data)                          #求样本均值

    if sigma is None:                               #方差未知
        s=std(data)                                 #求样本方差
        se = s / sqrt(n)                            #求标准误
        t_value=abs(t.ppf(alpha/2,n-1))             #求Z
        return sample_mean-se*t_value,sample_mean+se*t_value
    else:                                           #方差已知
        se=sigma/sqrt                               #求标准误
        z_value=abs(norm.ppf(alpha/2))              #求Z,由于取的Z alpha/2默认是返回坐标左边的面积，所以需要取绝对值
        return sample_mean-se*z_value,sample_mean+se*z_value

def var_ci_est(data,alpha):
    """
        总体均值未知，求方差的置信空间
        data为传入的样本; alpha为需要传入的置信水平的值
    """
    n = len(data)                                   #求样本容量
    s2 = variance(data)                             #求样本方差

    chi2_lower_value=chi2.ppf(alpha/2,n-1)          #求坐标左侧Z面积，没错你没看错，因为数学证明的过程中是以右侧为基准的，但是scipy是以左侧为基准的
    chi2_upper_value=chi2.ppf(1-alpha/2,n-1)        #求坐标右侧Z面积
    return (n-1)*s2/chi2_upper_value,(n-1)*s2/chi2_lower_value

if __name__ == '__main__':
    salary_18=[1484,785,1598,1366,1716,1020,1716,785,3113,1601]     #18岁月收入数据
    salary_35=[902,4508,3809,3923,4276,2065,1601,553,3345,2182]     #35岁月收入数据

    print(mean(salary_18))                          #平均月收入的点估计
    print(mean_ci_est(salary_18,0.05))              #平均月收入的区间估计
    print(mean(salary_35))                          #平均月收入的点估计
    print(mean_ci_est(salary_35,0.05))              #平均月收入的区间估计
    print()
    print(std(salary_18))                           #整体方差的点估计开根
    print(variance(salary_18))                      #整体方差的点估计(样本方差)
    print(var_ci_est(salary_18,0.05))               #区间估计
    print(std(salary_35))                           #整体方差的点估计开根
    print(variance(salary_35))                      #整体方差的点估计(样本方差)
    print(var_ci_est(salary_35,0.05))               #区间估计
    print()
