package kmeans;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;



/**
 * Kmeans算法java实现
 * @author pby 2018.11.22
 *
 * */
public class algorithm {

    private static double[] irislabel;
    private static double[][] irisdata;
    private static int Epoch=100;
    private static int Clusternum=3;//定义类簇个数为3
    private static List<double[]>ulist=new ArrayList<>(); //用于存放聚类中心向量
    private static List<List<double[]>>clusters=new ArrayList<>();//存在最终获取的类簇集合
    private static List<List<Double>>clusterslabels=new ArrayList<>();//存放类簇集合的label标签


    //从路径中加载数据
    private static void loadirisdata(){
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("data/iris/formatiris.txt"),"UTF-8"));
            int row=Integer.parseInt(br.readLine());
            int col=Integer.parseInt(br.readLine());
            //初始化全局数组iris
            irisdata=new double[row][col];
            irislabel=new double[row];
            String line=br.readLine();
            int linesnum=0;
            while (line!=null) {
                String[] temp = line.split("\\s");
                irislabel[linesnum]= Double.parseDouble(temp[0]);
                for (int i = 1; i < temp.length; i++) {
                    irisdata[linesnum][i-1] = Double.parseDouble(temp[i]);
                }
                linesnum++;
                line=br.readLine();
            }
            br.close();
            System.out.println("iris数据加载完成");
        }catch(Exception e){
            System.out.println("载入数据集错误");
        }
    }
    /**
     * 随机指定范围内N个不重复的数
     * @param min 指定范围最小值
     * @param max 指定范围最大值
     * @param n 随机数个数
     */
    public static int[] randomCommon(int min, int max, int n) {
        if (n > (max - min + 1) || max < min) {
            return null;
        }
        int[] result = new int[n];
        int count = 0;
        while (count < n) {
            int num = (int) (Math.random() * (max - min)) + min;//Math.random()值域为[0,1)的小数
            boolean flag = true;
            for (int j = 0; j < n; j++) {
                if (num == result[j]) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                result[count] = num;
                count++;
            }
        }
        return result;
    }
    /**
     * 计算两向量间的欧式距离
     * @param x1 array1
     * @param x2 array2
     * */
    private static double Euclidistance(double[] x1,double[] x2){
        double res=999.0;

        if (x1.length!=x2.length){
            System.out.println("检查输入向量长度是否不一致");
        }else{
            res=0;
            for (int i=0;i<x1.length;i++){
                res+=Math.abs(x1[i]-x2[i]);
            }
            res=Math.sqrt(res);
        }
        return res;
    }

    public static void kmeansCluster(int epoch){
        //初始化类簇集合
        for(int cl=0;cl<Clusternum;cl++){
            clusters.add(new ArrayList<>());
            clusterslabels.add(new ArrayList<>());
        }
        System.out.println("=====================>开始遍历第 "+epoch+" 个epoch<=================");
        double st = System.currentTimeMillis();
        for(int i=0;i<irisdata.length;i++){
            //最短距离类簇（不断更新）
            int minDisCluster=-1;
            //最短距离（不断更新）
            double minDis=-1.0;
            for(int j=0;j<ulist.size();j++){
                double Dis=Euclidistance(ulist.get(j),irisdata[i]);
                if(minDis>0&&Dis<minDis){
                    minDis=Dis;
                    minDisCluster=j;
                }else if(minDis<0){
                    minDis=Dis;
                    minDisCluster=j;
                }
           }
           //获取当前数据最接近的类簇
            clusters.get(minDisCluster).add(irisdata[i]);
            clusterslabels.get(minDisCluster).add(irislabel[i]);
        }
        System.out.println("======>共耗时 "+(double)(System.currentTimeMillis()-st)/1000.0d+"秒<====");

    }

    /**
     * 返回两个聚类中心向量集合间的的累计误差
     * */
    private static double loss( List<double[]>oldlist,List<double[]>newlist){
        double error=0.0;//统计两个欧几里得距离的累计误差
        for(int i=0;i<oldlist.size();i++){
            error+= Euclidistance(oldlist.get(i),newlist.get(i));
        }
        return  error;
    }

    /**
     * k均值聚类运行函数 带参数
     * @param epoch 终止遍历次数
     * @param clusternum 聚类簇的个数
     * */
    public static void kmeansrun(int epoch,int clusternum){
        Epoch=epoch;
        Clusternum=clusternum;
        kmeansrun();
    }


    /**
     * k均值聚类运行函数 无参数
     * */
    public static void kmeansrun(){
        //先加载数值数据
        loadirisdata();
        //依据设置的类簇个数对类簇集合进行初始化
        for(int k=0;k<Clusternum;k++){
            clusters.add(new ArrayList<>());
        }
        int[] index=randomCommon(0,irisdata.length,Clusternum);//随机获取三个聚类中心
        for(int u=0;u<index.length;u++){
            ulist.add(irisdata[index[u]]);//依次获取三个随机行向量作为初始平均值
        }
        int sepoch=0;
        while(sepoch<Epoch){
            //对老的聚类中心进行深复制
            List<double[]>ulistold=new ArrayList<>();
            for(double[] up:ulist){
                ulistold.add(up);
            }

            clusters.clear();
            clusterslabels.clear();

            kmeansCluster(sepoch);
            ulist.clear();
            //重新计算聚类中心
            for(List<double[]>C:clusters){
                double[] Ctotal=new double[4];//这里自己给！！
                for(double[] sample:C){
                    for(int i=0;i<sample.length;i++){
                        Ctotal[i]+=sample[i];
                    }
                }
                for(int c=0;c<Ctotal.length;c++){
                    Ctotal[c]=Ctotal[c]/C.size();
                }
                ulist.add(Ctotal);
            }
            //比较更新前后的聚类中心变化情况
            int clu=0;
            for(List C:clusters){
                System.out.println("类簇"+clu+"包含"+C.size()+"个向量集合");
                System.out.println("类簇"+clu+"真实标签集合为"+clusterslabels.get(clu));
                clu++;
            }

            double lossvalue=loss(ulistold,ulist);
            System.out.println("====>第"+sepoch+"个epoch结束后的均值向量距离差值为"+lossvalue);
            if(lossvalue<0.0001){
                System.out.println("达到收敛条件，提前中断迭代！");
                double A=0,B=0,C=0;
                for(double val:irislabel){
                    if(val==0.0){
                        A++;
                    }else if(val==1.0){
                        B++;
                    }else{
                        C++;
                    }
                }
                System.out.println("==========真实数据分布情况为===========");
                System.out.println("类A包含"+A+"个向量集合");
                System.out.println("类B包含"+B+"个向量集合");
                System.out.println("类C包含"+C+"个向量集合");

                break;
            }
            sepoch++;
        }
    }
    public static void main(String[] args){
        kmeansrun(5000,3);
    }

}
