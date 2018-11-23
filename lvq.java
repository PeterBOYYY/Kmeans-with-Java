package cluster;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * 学习向量量化法（LVQ）进行聚类算法实现
 * 带标签的辅助聚类算法
 * 2018.11.23
 * @author pby
 * */
public class lvq {
    private static double[] irislabel;
    private static double[][] irisdata;
    private static int Epoch=50000;
    private static double uLabels[]={0.0,1.0,2.0};//定义原型向量类别标记,数组形式
    private static int uSize=3;//定义原型向量个数
    private static double learningrate=0.0001;//更新学习率
    private static List<double[]>ulistold=new ArrayList<>();//保存上一次迭代的聚类原型向量
    private static List<double[]> ulist=new ArrayList<>(); //用于存放聚类中心向量
    private static List<List<double[]>>clusters=new ArrayList<>();//存在最终获取的类簇集合
    private static List<List<Double>>clusterslabels=new ArrayList<>();//存放类簇集合的label标签

    /**
     * 加载数据
     * */
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
        double res=-1;
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

    /**
     * 计算向量间的直接距离
     * @param x1 减向量（遍历向量）
     * @param x2 被减向量(原型向量)
     * */
    private static double[] distance(double[] x1,double[] x2){
        double[] res=new double[x1.length];
        for(int i=0;i<x1.length;i++){
            res[i]=x1[i]-x2[i];
        }
        return res;
    }

    /**
     * 返回两个聚类中心向量集合间的的累计误差
     * */
    private static Double loss( List<double[]>oldlist,List<double[]>newlist){
        DecimalFormat df = new DecimalFormat("0.000000");
        Double error=0.0000000;//统计两个欧几里得距离的累计误差
        for(int i=0;i<oldlist.size();i++){
            error+= Euclidistance(oldlist.get(i),newlist.get(i));
        }
        return  Double.parseDouble(df.format(error));
    }



    /**
     * @param Epochs 最大迭代次数
     * @param Csize 原型向量类别个个数
     * @param Clabel 原型向量初始化类别
     *
     * */
    public static void lvqrun(int Epochs,int Csize,double[] Clabel){
        Epoch=Epochs;
        uSize=Csize;
        uLabels=Clabel;
        lvqrun();

    }


    /**
     *
     * 使用序列化方法对list进行深拷贝
     * */
    public static <T> List<T> deepCopy(List<T> src) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(byteOut);
        out.writeObject(src);

        ByteArrayInputStream byteIn = new ByteArrayInputStream(byteOut.toByteArray());
        ObjectInputStream in = new ObjectInputStream(byteIn);
        @SuppressWarnings("unchecked")
        List<T> dest = (List<T>) in.readObject();
        return dest;
    }
    /**
     * 聚类运行接口
     * */
    public static void lvqrun(){
        //加载数据
        loadirisdata();
        int[] index=randomCommon(0,irisdata.length,uSize);//随机获取三个原型向量在原始数据集的索引
        for(int u=0;u<index.length;u++){
            ulist.add(irisdata[index[u]]);//依次获取三个原型向量，每个向量的标签与ulabel中一一对应
        }
        int sEpoch=0;//定义初始迭代次数
        while(sEpoch<Epoch){
            //对老的聚类中心进行深复制
            ulistold.clear();
            try {
                ulistold = deepCopy(ulist);
            }catch (Exception e){
                System.out.println("深拷贝列表失败");
            }


            lvqfunction(sEpoch);//执行聚类功能函数
            //比较更新前后的聚类中心变化情况
            int clu=0;
            for(List C:clusters){
                System.out.println("类簇"+clu+"包含"+C.size()+"个向量集合");
                System.out.println("类簇"+clu+"真实标签集合为"+clusterslabels.get(clu));
                clu++;
            }

            double lossvalue=loss(ulistold,ulist);
            System.out.println("====>第"+sEpoch+"个epoch结束后的均值向量距离差值为"+lossvalue);
            if(lossvalue<0.00000001){
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

            sEpoch++;
        }


    }

    /**
     * 对数据集使用lvq进行一次遍历
     * @param tepoch 当前epoch
     * */
    public static void lvqfunction(int tepoch){
        clusters.clear();
        clusterslabels.clear();

        //初始化类簇集合
        for(int cl=0;cl<uSize;cl++){
            clusters.add(new ArrayList<>());
            clusterslabels.add(new ArrayList<>());//同步初始化类簇集合下的真实标签集合
        }

        System.out.println("=====================>开始遍历第 "+tepoch+" 个epoch<=================");
        double st = System.currentTimeMillis();
        for(int i=0;i<irisdata.length;i++){
            //最短距离类簇（不断更新）
            double minDisulabel=-1;
            //最短距离（不断更新）
            double minDis=-1.0;
            //最接近的原型向量对应位置
            int minuidx=-1;
            for(int j=0;j<ulist.size();j++){
                double Dis=Euclidistance(ulist.get(j),irisdata[i]);
                if(minDis>0&&Dis<minDis){
                    minDis=Dis;
                    minDisulabel=uLabels[j];
                    minuidx=j;
                }else if(minDis<0){
                    minDis=Dis;
                    minDisulabel=uLabels[j];
                    minuidx=j;
                }
            }
            //这里对原型向量进行更新
            if(minDis>0){
                /*
                判断与所分配类簇的原型向量标签是否相同
                相同则让原型向量向被分配向量的欧式空间距离靠近
                不同则远离
                */
                //获取对应原型向量
                double[] u=ulist.get(minuidx);
                double[] loss=distance(irisdata[i],u);
                if(irislabel[i]==minDisulabel){
                    for(int ui=0;ui<u.length;ui++){
                        u[ui]=u[ui]+learningrate*loss[ui];
                    }
                    ulist.set(minuidx,u);//在ulist中对minuidx位置的u进行更新
                }else{
                    for(int ui=0;ui<u.length;ui++){
                        u[ui]=u[ui]-learningrate*loss[ui];
                    }
//                    ulist.set(minuidx,u);//在ulist中对minuidx位置的u进行更新
                }
            }
            //获取当前数据最接近的类簇
            clusters.get((int)minDisulabel).add(irisdata[i]);
            clusterslabels.get((int)minDisulabel).add(irislabel[i]);
        }
        System.out.println("======>共耗时 "+(double)(System.currentTimeMillis()-st)/1000.0d+"秒<====");

    }



    public static void main(String[] args){
        lvqrun();
    }




}
