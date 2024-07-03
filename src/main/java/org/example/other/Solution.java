package org.example.other;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 返回a+b的和
     * @param a int整型
     * @param b int整型
     * @return int整型
     */
    public int add (int a, int b) {
        // write code here
        return a+b;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     *
     * @param k int整型
     * @param target int整型
     * @return bool布尔型
     */
    public boolean kInArray (int k, int target) {
        return target==k||target-k-k==1||target-k-k-k==1;
    }
    public boolean kInArray (long k, long target) {
        return target==k||target-k==k+1||target-k-k==k+1;
    }

    public int[] flipImage (int width, int[] pixels) {
        int row= pixels.length/width/4;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < width/2; j++) {
                int index=i*width+j,toI=(i+1)*width-j-1;
                change(pixels,index,toI,4 );
            }
        }
        return pixels;
    }

    private void change(int[] arr,int start,int to,int len){
        int[] temp=new int[len];
        start*=4;
        to*=4;
        System.arraycopy(arr,start,temp,0,len);
        System.arraycopy(arr,to,arr,start,len);
        System.arraycopy(temp,0,arr,to,len);
    }

    public static void main(String[] args) {
        new Solution().flipImage(2,new int[]{1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4});
    }
}