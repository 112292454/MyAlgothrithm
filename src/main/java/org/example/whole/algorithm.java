package org.example.whole;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.System.*;


public class algorithm {
	static int[] dx = {1,-1,0,0,1,1,-1,-1};
	static int[] dy = {0,0,1,-1,1,-1,1,-1};
	static long e97=1000000007;
	public static void main(String args[]) {
		Scanner in=new Scanner(System.in);
		//while(in.hasNext()){
			try {
				int n=in.nextInt();
				if(n<1||n>10) throw new RuntimeException("WRONG!");
				for (int i = 0; i < n; i++) {
					int a=in.nextInt(),b=in.nextInt();
					if(a>1018||b>1018||a<0||b<0) throw new RuntimeException("WRONG!");
					if (Math.abs(a - b) > 1||(a==0&&b==0)) out.println("GG");
					else out.println("MM");
				}
			}catch (Exception exception){
				out.println("Input data error");
			}

		//}
		/*
		String[] strs=new String[n];
		for (int i = 0; i < n; i++) strs[i]=in.nextLine();
		String find=in.nextLine();
		Arrays.sort(strs);
		for (String str : strs) {
			if(str.startsWith(find)) System.out.println(str);
		}*/

	}
	static void change(int[] a, int x, int y){
		int k=a[y];
		for(int i=y;i>x;i--){
			a[i]=a[i-1];
		}
		a[x]=k;
	}
	static String out1(int[] a){
		StringBuilder sb=new StringBuilder();
		sb.append(a[0]);
		for (int i = 1; i < a.length; i++) {
			sb.append(" ").append(a[i]);
		}
		sb.append("\n");
		return sb.toString();
	}
	static boolean  check(int[][] a,int x,int y){
		return x>=0&&y>=0&&x<a.length&&y<a.length;
	}
}


class Reader{
	private final static int BUF_SZ = 65536;
	BufferedReader in;
	StringTokenizer tokenizer;
	public Reader(InputStream in) {
		super();
		this.in = new BufferedReader(new InputStreamReader(in),BUF_SZ);
		tokenizer = new StringTokenizer("");
	}

	public String next() {
		while (!tokenizer.hasMoreTokens()) {
			try {
				tokenizer = new StringTokenizer(in.readLine());
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken();
	}
	public int nextInt() {
		return Integer.parseInt(next());
	}
	public long nextLong() {
		return Long.parseLong(next());
	}
}

class node{
	//public int value,x;
	public int father,son,top;//可能变为node对象
	public int depth,treeSize,value,id;//数值
	public int x,y;//坐标
	@Override
	public node clone(){
		node t=new node();
		t.father=father;t.son=son;t.top=top;
		t.depth=depth;t.treeSize=treeSize;t.value=value;t.id=id;
		t.x=x;t.y=y;
		return t;
	}

	public node(int value, int x, int y) {
		this.value = value;
		this.x = x;
		this.y = y;
	}

	public node() {

	}
}

/*
		long[][] matrix={{1,1},{1,0}};
		long[] a={1,1};
		matrix=ot.matrixPow(matrix,1145141919810L-1,998244353);
		out.println("m = " + Arrays.toString(matrix));

		for (int i = 0; i < 100; i++) {
			int res=test(8,i);
			out.println("if b="+i+"  thus,a = " + res);
		}
		Reader in = new Reader(System.in);
		//int times=in.nextInt(),t=0;
		//Pattern p2=Pattern.compile("(\\W+)|(\\w+)");
		//l:while (in.hasNext())
		//long cntTime= currentTimeMillis();
		StringBuilder sb=new StringBuilder();
		//l:for (int i = 0; i < times; i++) {
		int n=in.nextInt(),m=in.nextInt(),index,indexu,s,e,l,t;
		int[] arr=new int[n];
		HashMap<Integer,Boolean> h=new HashMap<>(200000);
		while (n-->0) {arr[n]=in.nextInt();h.put(arr[n],true);}
		Arrays.sort(arr);
		n=arr.length;
		for (int i = 0; i < m; i++) {
			t=in.nextInt();
			indexu=index=ot.lowerBound(arr,t);
			if(h.containsKey(t))
				indexu=ot.upperBound(arr,t);
			s=index;
			l=0;
			e=0;
			if(index==-1) s=n;
			else if(indexu==-1) e=n-index;
			else {e=indexu-index;l=n-indexu;}
			sb.append(s+" "+e+" "+l+"\n");
		}
		//out.println(s2 == s1);
		//out.println(s == s2);
		//if(i>50000&& i%1000==0) out.print(t - System.currentTimeMillis()+"  ");
		//}
		out.println(sb);
		//long t=System.currentTimeMillis();
		//System.out.println(t - System.currentTimeMillis());*/

class OtherTool{
	public static int gcd(int a, int b){return b==0?a:gcd(b,a%b);}

	//矩阵转置
	public static long[][] MatrixTranspose(long[][] matrix){
		long len=matrix.length,temp=0;
		if(len!=matrix[0].length) throw new IllegalArgumentException();
		for (int i = 0; i < len; i++) {
			for (int j = i; j < len; j++) {
				temp=matrix[i][j];
				matrix[i][j]=matrix[j][i];
				matrix[j][i]=temp;
			}
		}
		return matrix;
	}
	//矩阵乘法
	public static long[][] matrixMul(long[][] matrix1,long[][] matrix2, long mod) {
		int row = matrix1.length, column = matrix2[0].length, max = max(row, column), cnt1 = 0, cnt2 = 0;
		long[][] temp = new long[row][column];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				long res = 0;
				for (int k = 0; k < max; k++) {
					//if(matrix1[i][k]==1) cnt1++;
					//if(matrix2[k][j]==1) cnt2++;
					res += matrix1[i][k] * matrix2[k][j];
					res %= mod;
					//if(cnt1>1&&cnt2>1) {cnt1=cnt2=0;break;}
				}
				temp[i][j] = res % mod;
			}
		}
		return temp;
	}
	//矩阵幂次——快速幂
	public static long[][] matrixPow(long[][] matrix,long times, long mod){
		if(times==1) return matrix;
		int l= matrix.length;
		long[][] temp=new long[l][l];
		for (int i = 0; i < l; i++) {
			temp[i][i]=1;
		}
		if(times<1) return temp;
		while (times>0){
			if(times%2==1) temp=matrixMul(temp,matrix,mod);
			matrix=matrixMul(matrix,matrix,mod);
			times/=2;
		}
		return temp;
	}
	//数字快速幂
	public static long fastPower(int source, long times, long mod){
		if(times==0||source==1) return 1;
		long i=1,cnt=1;
		while (times>0){
			if(times%2==1) i=(source*i)%mod;
			source*=source;
			source%=mod;
			times/=2;
		}
		return i;
	}

	//快速排序
	/*
	def quick_sort(array, l, r):
  if l < r:
    q = partition(array, l, r)
    quick_sort(array, l, q - 1)
    quick_sort(array, q + 1, r)

def partition(array, l, r):
  x = array[r]
  i = l - 1
  for j in range(l, r):
    if array[j] <= x:
      i += 1
      array[i], array[j] = array[j], array[i]
  array[i + 1], array[r] = array[r], array[i+1]
  return i + 1
*/

	//意义不大，可以写成对对象排序，保留，作为归并排序的参考写法
	public static void twoArraySort(int[] a, int[] b, int l, int r) {

		int mid = (l + r) / 2;
		if (l < r - 1) {
			twoArraySort(a, b, l, mid);
			twoArraySort(a, b, mid + 1, r);
		}
		if (l == r) return;
		int[] temp = new int[r - l + 1];
		int[] tempb = new int[r - l + 1];
		int i = l, j = mid + 1, t = 0;
		while (i < mid + 1 && j < r + 1) {
			if (a[i] <= a[j]) {
				temp[t] = a[i];
				tempb[t++] = b[i++];
			} else {
				temp[t] = a[j];
				tempb[t++] = b[j++];
			}
		}
		while (i < mid + 1) {
			temp[t] = a[i];
			tempb[t++] = b[i++];
		}
		while (j < r + 1) {
			temp[t] = a[j];
			tempb[t++] = b[j++];
		}
		for (int k = l; k < r + 1; k++) {
			a[k] = temp[k - l];
			b[k] = tempb[k - l];
		}
	}
	//对升序数组二分查找,大于等于target的第一个数下标，均大于时返回0，均小于时返回-1
	public static long lowerBound(long[] a,long target){
		long l=0,r=0,mid,half,len=a.length-1;
		for (long l1 : a) {
			r=max(l1,r);
		}
		len=r;
		while (len>0){
			half=len/2;
			mid=l+half;//普通二分即把boundv改成a【i】
			if(boundValue(a,mid)<=target) {
				l=mid+1;
				len=len-half-1;
			}else len=half;
		}
		if(target>boundValue(a,l)) return -1;
		return l;
	}
	public static int lowerBound(int[] a,int target){
		int l=0,r=0,mid,half,len=a.length-1;
		while (len>0){
			half=len/2;
			mid=l+half;//普通二分即把boundv改成a【i】
			if(a[mid]<target) {
				l=mid+1;
				len=len-half-1;
			}else len=half;
		}
		if(target>a[l]) return -1;
		return l;
	}
	public static int upperBound(int[] a,int target){
		int l=0,r=0,mid,half,len=a.length-1;
		while (len>0){
			half=len/2;
			mid=l+half;//普通二分即把boundv改成a【i】
			if(a[mid]<=target) {
				l=mid+1;
				len=len-half-1;
			}else len=half;
		}
		if(target>=a[l]) return -1;
		return l;
	}
	private static long boundValue(long a[],long t){
		long res=0;
		for (long l : a) {
			if(l<t) res+=l;
			else res+=t;
		}
		return res;
	}
	//字符串的下一个全排列
	public static String nextPermutation(String s){
		if(s.length()<=1) return null;
		StringBuilder sb=new StringBuilder(s);
		int i1=-1,i2=-1;
		for (int i = s.length()-1; i >=0 ; i--) if(s.charAt(i-1)<s.charAt(i)) {i1=i-1;break;}
		if(i1==-1) return null;
		for (int i = s.length()-1; i >=0 ; i--) if(s.charAt(i)>s.charAt(i1)) {i2=i;break;}
		char c=s.charAt(i1);
		sb.setCharAt(i1,s.charAt(i2));
		sb.setCharAt(i2,c);
		sb.replace(i1,s.length(),new StringBuilder(sb.substring(i1,s.length())).reverse().toString());
		return sb.toString();
	}
	//数组的下一个全排列
	public static int[] nextPermutation(int[] s){

		if(s.length<=1) return null;
		int i1=-1,i2=-1;
		for (int i = s.length-1; i >0 ; i--) if(s[i-1]<s[i]) {i1=i-1;break;}
		if(i1==-1) return null;
		for (int i = s.length-1; i >=0 ; i--) if(s[i]>s[i1]) {i2=i;break;}
		int t=s[i1];
		s[i1]=s[i2];
		s[i2]=t;
		t=s.length-i1-1;
		int[] temp=new int[t];
		for (int i = 0; i < t; i++) {
			temp[i]=s[s.length-i-1];
		}
		arraycopy(temp,0,s,i1+1,t);
		return s;
	}
	/*//康托展开，将排列化为数字（实为该排列在其全排列中的顺序，排序长度不可超过20
	public static long cantorExpansion(int[] arr){
		if(arr.length>=20) return -1;
		HashSet<Integer> used=new HashSet<>();
		HashMap<Integer,Integer> little=new HashMap<>();
		long res=0;

	}
	//康托逆展开，还原为一个排序
	public static int[] RECantorExpansion(long l){

	}
	*/
	//欧拉筛，线性
	public static int[] primes(int range){

		double nums=0;
		if(range>1000000) nums=range*(Math.log(range)+0.01);
		else nums=80000;
		int[] res=new int[(int) nums];
		int t=1;
		boolean[] vis=new boolean[range];
		vis[1]=true;
		for (int i = 2; i < range; i++) {
			if(!vis[i]) res[t++]=i;
			for (int j = 1; j < t&&i*res[j]<range; j++) {
				vis[res[j]*i]=true;
				if(i%res[j]==0) break;
			}
		}
		return Arrays.copyOfRange(res,1,t);
	}
	//求出kmp预处理的next数组
	public static int[] KMPBuildNext(String s1,String s2){
		//实质是对于i，得到s1长i的前缀与s2长i的后缀，两者能够相同的最长长度
		//当s1=s2时即为求一个串最长的公共前后缀——为kmp的next数组

		int len= max(s1.length(),s2.length())+1,i=1,t=1;
		int[] res=new int[len];
		char[] c1=new char[len],c2=new char[len];
		for (char c:s1.toCharArray()) c1[i++]=c;
		i=1;
		for (char c:s2.toCharArray()) c2[i++]=c;
		int j=0;

		for (i = 0; i < len-1; i++) {
			while(j>0&&c1[i+1]!=c2[j+1]) j=res[j];
			if(c1[i+1]==c2[j+1]&&j<=i-1) j++;
			res[t++]=j;
		}
		return res;
	}
	private static int[] KMPBuildNext(String s){return KMPBuildNext(s,s);}
	//KMP
	public static int KMP(String source,String needle){
		if(needle.length()>source.length()) return -1;
		if(needle.length()==0) return 0;
		int[] next=KMPBuildNext(needle);
		int la=source.length(),lb=needle.length();
		char[] c1=("0"+source).toCharArray(),c2=("0"+needle).toCharArray();
		for (int i = 0,j=0; i < la; ) {
			while(j>0&&c1[i+1]!=c2[j+1]) j=next[j];
			while(j==0&&c1[i+1]!=c2[j+1]) {
				if(i==la-1) return -1;
				i++;
			}
			i++;j++;
			if(j==lb) return i-j;
		}
		return -1;
	}
	//判断数字是否同号——有为0也可以
	public static boolean isSameSymbol(double a,double b){
		return (a>=0&&b>=0)||(a<=0&&b<=0);
	}
	//得到数字的二进制表达中0的数量(正数)
	public static int getBinaryOneCount(long a){
		int cnt=0;
		while(a!=0) {a&=a-1;cnt++;}
		return cnt;
	}
	public static int getBinaryOneCount(int a) {return getBinaryOneCount((long)a);}
	//两点距离
	public static double distance(double x1,double y1,double x2,double y2){
		return Math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
	}
	//传入点集，返回凸包数组
	public static node[] convexHull(node[] arr){
		node[]res=new node[arr.length];
		int p=0,top=2;
		for (int i = 1; i < arr.length; i++) {
			if(arr[i].y<arr[p].y) p=i;
			else if(arr[i].y==arr[p].y&&arr[i].x<arr[p].x) p=i;
		}
		node o=arr[p];
		Arrays.sort(arr, ( o1,  o2) -> {
			double d=new vector(o.x,o.y,o1.x,o1.y).
					CrossProduct(new vector(o.x,o.y,o2.x,o2.y));
			if(d>0) return -1;
			else if(d<0) return 1;
			else return distance(o.x,o.y,o1.x,o1.y)<distance(o.x,o.y,o2.x,o2.y)?-1:1;
		});
		res[0]=arr[0];
		if(arr.length>1) res[1]=arr[1];
		if(arr.length>2) res[2]=arr[2];
		for (int i = 3; i < arr.length; i++) {
			while (new vector(res[top-1].x,res[top-1].y,res[top].x,res[top].y).
					CrossProduct(new vector(res[top].x,res[top].y,arr[i].x,arr[i].y))<0&&top>1) top--;
			res[++top]=arr[i];
		}
		return Arrays.copyOf(res, min(top+1,arr.length));
	}
}
class ShortestRoad extends AdjacencyTable {
	public ShortestRoad(int i) {
		super(i);
	}
	double[][] f;
	public void setFloyd(double[][] f) {
		this.f=f;
	}
	public int dijkstra(int start ,int end){
		int[] distance=new int[node.length],used=new int[node.length];
		PriorityQueue<Node> node=new PriorityQueue<Node>(Comparator.comparingInt(o -> o.value));
		Arrays.fill(distance,Integer.MAX_VALUE/2);
		node.add(new Node(start,0));
		distance[start]=0;
		while (!node.isEmpty()) {
			int city = node.poll().to;
			if (used[city] != 0) continue;
			used[city] = 1;
			ArrayList<Node> arr = map[city];
			if (arr.isEmpty()&&city!=end) return -1;
			for (Node n : arr) {
				int nowcity = n.to;
				if (distance[nowcity] > distance[city] + n.value) {
					distance[nowcity] = distance[city] + n.value;
					node.offer(new Node(nowcity, distance[nowcity]));
				}
			}
		}
		if(distance[end]!=Integer.MAX_VALUE/2) return distance[end];
		else return Integer.MAX_VALUE;//若返回maxint，则为不连通
	}
	public void floydRun(int max) {
		for (int k = 0; k < max; k++)
			for (int i = 0; i < max; i++)
				if (f[i][k] != Integer.MAX_VALUE / 2)
					for (int j = 0; j < max; j++) {
						if (f[k][j] != Integer.MAX_VALUE / 2) {
							f[i][j] = max(f[i][j], f[i][k] * f[k][j]);
							//f[j][i] = min(f[j][i], f[i][k] + f[k][j]);
						}
					}
	}
	public double floydAsk(int i,int j){
		return f[i][j];
	}
}
class segTree{
	/*
	疑似常数过大/写假了，之后需尝试用segment【】数组重写一遍，看看能不能降到可接受的时间
	现在貌似比ac时间多了几倍
	//已完成
	 */
	class segment{
		int l,r,lazy=0;
		int value=0;
		public segment(int a,int b){l=a;r=b;}
	}
	segment[] seg;
	node[] source;
	int size=0;
	public segTree(node[] source){
		size=source.length;
		seg=new segment[source.length<<2];
		this.source=source;
		build(1,size,1);
	}
	private void build(int l,int r,int index){
		if(seg[index]==null) seg[index]=new segment(l,r);
		if(l==r) {
			segment s=new segment(l,r);
			s.value=source[l-1].value;
			seg[index]=s;
		}
		else {
			int mid=(l+r)/2;
			build(l,mid,index<<1);
			build(mid+1,r,index<<1|1);
			pushUp(index);
		}
	}
	private void pushUp(int a){
		seg[a].value= max(seg[a<<1].value,seg[a<<1|1].value);
		//a.value=seg[a<<1.value+a.right.value;
	}
	private void pushDown(int a){
		if(seg[a].lazy!=0){
			//seg[a<<1].value=seg[a].lazy*seg[a<<1].length;
			//seg[a<<1|1].value=seg[a].lazy*seg[a<<1|1].length;
			/*处理a的两个子节点的value变化，视实现而定*/
			seg[a<<1].lazy=seg[a<<1|1].lazy=seg[a].lazy;
			seg[a].lazy=0;
		}
	}
	public void rangeModify(int start,int end,int value){
		rangeModify(1,start,end,value);
	}
	public long rangeQuery(int start, int end){
		return rangeQuery(1,start,end);
	}
	private void rangeModify(int i,int start,int end,int value){
		int l=seg[i].l,r=seg[i].r,mid=(l+r)/2;
		if(l>=start&&r<=end) {
			//a.value += value * (r - l + 1);
			if(seg[i].value<value) {
				seg[i].value = value;
				//seg[i].lazy = value;
			}
			//视实现而定
			return;
		}else if(l>end||r<start||r==l) return;
		//pushDown(i);
		if(mid>=start) rangeModify(i<<1,start,end,value);
		if(mid<=end) rangeModify(i<<1|1,start,end,value);
		pushUp(i);
	}
	private long rangeQuery(int i, int start, int end){
		int l=seg[i].l,r=seg[i].r;
		if(l>=start&&r<=end) {
			return seg[i].value;
		}else if(r<start||l>end) return 0;
		//pushDown(i);
		return max(rangeQuery(i<<1,start,end),rangeQuery(i<<1|1,start,end));
	}
}
class DisJointSetMapVersion {
	HashMap<Integer,Integer> parent, rank;
	int unConnectNum;

	public DisJointSetMapVersion(int size) {
		parent = new HashMap<>(size);
		rank = new HashMap<>(size);
		unConnectNum = size;
	}
	public int add(int i){
		if(!parent.containsKey(i)) {
			parent.put(i,i-1);
			if(parent.containsKey(i-1)) union(i,i-1);
			return i;
		}else {
			int blank=find(i);
			if(blank<=0) return 0;
			parent.put(blank,blank-1);
			return blank;
		}

	}
	public int find(int a) {
        /*if (a > parent.size()|| a < 0) {
            throw new IllegalArgumentException("FindIndexOutOfBounds:" + a);
        }*/
		while (parent.containsKey(a)&&!parent.get(a).equals(a)) {
			//路径压缩，两步一跳
			if(parent.containsKey(parent.get(a)))
				parent.replace(a,parent.get(parent.get(a)));
			if(!parent.containsKey(a)) parent.put(a,a);
			a = parent.get(a);
		}
		//parent.remove(a);
		return a;
	}

	public boolean isConnect(int a, int b) {
		return find(a) == find(b);
	}

	public int union(int a, int b) {
		//按秩合并，仅当相同时高度+1；find会改变秩，仅作参考，是否会造成错误待定
		int aroot = find(a);
		int broot = find(b);
		if (aroot == broot) return aroot;
		unConnectNum--;
		if (rank.get(aroot) > rank.get(broot)) {
			parent.replace(broot,aroot);
			return aroot;
		} else if (rank.get(aroot) < rank.get(broot)) {
			parent.replace(aroot,broot);
			return broot;
		} else {
			parent.replace(broot,aroot);
			rank.replace(aroot,rank.get(aroot+1));
			return aroot;
		}
	}
}
class ot extends OtherTool{}
//树状数组
class BinaryIndexTree{
	long[] tree1,tree2;
	int size;
	public BinaryIndexTree(int size) {
		tree1=new long[100008];
		tree2=new long[100008];
		this.size=size;
	}
	public void add(int index,long value){
		long p=index*value;
		while (index<=size) {
			tree1[index]+=value;
			tree2[index]+=p;
			index+=index&-index;
		}
	}
	public void RangeAdd(int start,int end,int value){
		add(start,value);
		//if(end<100003)
		add(end+1,-value);
	}
	public long query(int start,int end){
		return sum(end)-sum(start-1);
	}
	public long sum(int i){
		long res=0,p=i;
		while (i>0) {
			res+=tree1[i]*(p+1)-tree2[i];
			i-=i&-i;
		}
		return res;
	}
}

/*
//待修改邻接表的实现方式——即下面最短路的方式
class BipartiteGraph  extends   AdjacencyTable{
    ConnectRelationNode[] map;
    int[] matched;
    boolean[] used;
    public BipartiteGraph(int i) {
        map=new ConnectRelationNode[i+1];
        used=new boolean[i+1];matched=new int[i+1];
        Arrays.fill(matched,-1);
        for (int i1 = 0; i1 < map.length; i1++) {
            map[i1]= new ConnectRelationNode();
        }
    }
    @Override
    public void setRoad(int start, int next, int value){
        ConnectRelationNode c=new ConnectRelationNode(next,value);
        c.nextConnectNode=map[start].nextConnectNode;
        map[start].nextConnectNode=c;
    }
    public int Hungarian(){
        int maxWays=0;
        for (int i = 0; i < map.length; i++) {
            Arrays.fill(used,true);
            if(map[i].nextConnectNode==null)continue;
            if(canMatch(i)) maxWays++;
        }
        return maxWays;
    }
    public boolean canMatch(int from){
        ConnectRelationNode n=map[from].nextConnectNode;
        int to=n.nowConnectNode;
        while (!used[to]&&n.nextConnectNode!=null) {n=n.nextConnectNode;to=n.nowConnectNode;}
        if(used[to]){
            used[to]=false;
            if(matched[to]==-1||canMatch(matched[to])){
                matched[to]=from;
                return true;
            }
        }
        return false;
    }
}*/
class treeToLink extends AdjacencyTable{
	/*
	map：树的链接关系，双向
	数组:id——树上节点的下标
		father——节点的父节点
		son——节点的重儿子节点
		depth——节点的深度
		size——以节点为根的子树大小——在newId中可通过node+size【node】确定一颗子树的所有点
		第一遍dfs
		top——节点所在重链的根节点
	 以下如果不建线段树可以不求（仅有lca）
		newId——树节点序号作为下标，存储其dfs序——保证重链、子树上的节点在此数组中对应的值连续
		oldID——以dfs序为下标，存储树节点
	seg：存树节点进行区间操作的线段树
		对oldId建线段树，这样一条重链/子树上的点作为下标所对应newId的值，在oldId中是连续的一段区间，可以进行区间操作
	 */
	node[] s,dfsTree;
	int[] dfsId;
	public treeToLink(int i) {
		super(i);
		s=new node[i];
		//for (int j = 0; j < s.length; j++) addNode(j);
	}
	public void addNode(int id,int value){
		node n=new node();
		n.id=id;
		n.value=value;
		n.son=0;
		n.treeSize=1;
		s[id]=n;
	}
	public void addNode(int id){addNode(id,1);}

	public void start(){
		addNode(0);
		dfs1(1,0,1);
		dfs2(1,1);
	}
	private void dfs1(int nowId,int fId,int depth){
		s[nowId].father=fId;
		s[nowId].treeSize=1;
		s[nowId].depth=depth;
		for (Node road : map[nowId]) {
			if(road.to==fId) continue;
			dfs1(road.to,nowId,depth+1);
			s[nowId].treeSize+=s[road.to].treeSize;
			if(s[road.to].treeSize>s[s[nowId].son].treeSize) s[nowId].son=road.to;
		}
	}
	private void dfs2(int nowId,int top){
		s[nowId].top=top;
		if(s[nowId].son!=0) dfs2(s[nowId].son,top);
		for (Node road : map[nowId]) {
			if(road.to!=s[nowId].son&&road.to!=s[nowId].father) dfs2(road.to,road.to);
		}
	}
	public int lca(int a,int b){
		node x=s[a],y=s[b];
		while (x.top!=y.top){
			if(x.depth>=y.depth) x=s[s[x.top].father];
			else y=s[s[y.top].father];
		}
		if(x.depth<y.depth) return x.id;
		else return y.id;
	}
}
class StronglyConnectComponent extends AdjacencyTable{
	ArrayList<AdjacencyTable.Node>[] reMap;
	boolean[] used1,used2;
	int[] stack,component;
	int stackPtr=0,componentNum=0;
	public StronglyConnectComponent(int i) {
		super(i);
		reMap=new ArrayList[i];
		for (int i1 = 0; i1 < map.length; i1++) reMap[i1]= new ArrayList<>();
		used1=new boolean[i];
		used2=new boolean[i];
		stack=new int[i];
	}
	@Override
	public void setRoad(int start, int next, int value){
		map[start].add(new Node(next,value));
		reMap[next].add(new Node(start,value));
	}
	public void getComponent(){
		for (int i = 0; i < size; i++) {
			if(!used1[i]) dfsStack(i);
		}
		component=new int[size];
		for (int i = stackPtr-1; i >= 0; i--) {
			if(!used2[stack[i]]) dfsResult(stack[i],i);
			componentNum++;
		}
	}
	public void dfsStack(int nowNode){
		//得到后续当作无向图dfs求连通分量时的顺序
		if(!used1[nowNode]) used1[nowNode]=true;
		else return;
		for (Node n : reMap[nowNode]) {
			if(!used1[n.to]) dfsStack(n.to);
		}
		stack[stackPtr++]=nowNode;
	}
	public void dfsResult(int nowNode,int index){
		if(!used2[nowNode]) used2[nowNode]=true;
		else return;
		for (Node n : map[nowNode]) {
			if(!used2[n.to]) dfsResult(n.to,index);
		}
		component[nowNode]=index;
	}
	public int[] getComponentNum(){
		if(stackPtr==0) getComponent();
		int[] res=new int[stackPtr];
		int t=0;
		for (int i = 0; i < component.length; i++) res[component[i]]++;
		for (int re : res) {
			if(re!=0) res[t++]=re;
		}
		return Arrays.copyOf(res,t);
	}
}
class Pack {
	//含01、完全、多重背包，混合背包可并入多重,二维待定
	int[] weight, value, multiple;
	int nums;

	public Pack(int nums) {
		weight = new int[nums];
		value = new int[nums];
	}

	public Pack(int[] w, int[] v) {
		weight = w;
		value = v;
		nums = weight.length;
	}

	public Pack(int[] w, int[] v, int[] m) {
		weight = w;
		value = v;
		multiple = m;
		nums = weight.length;
	}

	public int ZeroOnePack(int volume) {
		//dp[i] = Math.max(dp[i-weight[k]]+value[k], dp[i]);
		//dp【i】为容积使用i时的最大价值，考虑前k种物品的部分由循环实现
		//确保没有物品放入多次，i从大到小————由没有选择k物品的状态转移而来
		int[] dp = new int[volume + 8],hash=new int[9003];
		for (int k = 0; k < nums; k++) {
			for (int i = volume; i > 0; i--) {
				hash[max(0,dp[i])]=1;
				if(i>=weight[k]) hash[max(0,dp[i - weight[k]] + value[k])]=1;
				if (i >= weight[k]) dp[i] = max(dp[i - weight[k]] + value[k], dp[i]);

			}
		}
		return dp[volume];
	}
	public long ZeroOnePackCount(int v){
		//得到总价值为value时选择的方案数，容量不做限制
		//dp【j】【i】为考虑前j个物品，价值为i时的方案数，j维度通过滚动for删除
		//dp[i]=i<value[j]?  dp[i] : dp[i]+dp[i-value[j]]
		//与上面相同，应倒序，保证dpi是从没有选择过物品j的状态转移而来
		long[] dp=new long[v];
		dp[0]=1;
		for (int j = 0; j < nums; j++) {
			for (int i = v; i >= 0; i--) {
				dp[i]=i<value[j]?dp[i]:dp[i]+dp[i-value[j]];
			}
		}
		return dp[v];
	}
	public int CompletePack(int volume) {
		//与01的差别为i从小到大————每个状态可以由已经选中了k物品的状态转移而来
		int[] dp = new int[volume + 1];
		for (int k = 0; k < nums; k++) {
			for (int i = weight[k]; i < volume + 1; i++) {
				if (i >= weight[k]) dp[i] = max(dp[i - weight[k]] + value[k], dp[i]);
			}
		}
		return dp[volume];
	}

	public int MultiplePack(int volume) {
		//多重背包，将每种物品可选的数量二进制化：用1/2/4/8……拼出，化为O(lgn)个新的物品
		ArrayList<Integer> w = new ArrayList<>();
		ArrayList<Integer> v = new ArrayList<>();
		for (int i = 0; i < multiple.length; i++) {
			int k = multiple[i]&-multiple[i];
			while (true) {
				if (k <= multiple[i]) {
					w.add(i, k * weight[i]);
					v.add(i, k * value[i]);
					multiple[i] -= k;
					k *= 2;
				} else {
					k = multiple[i];
					if(k!=0) {
						w.add(i, k * weight[i]);
						v.add(i, k * value[i]);
					}
					break;
				}
			}
		}
		int[] w1 = new int[w.size()], v1 = new int[v.size()];
		for (int i = 0; i < w1.length; i++) {
			w1[i] = w.get(i);
			v1[i] = v.get(i);
		}
		weight = w1;
		value = v1;
		nums = weight.length;
		return ZeroOnePack(volume);
	}
}

class DisJointSet {
	private int[] parent, rank;
	private int unConnectNum;

	public DisJointSet(int size) {
		parent = new int[size];
		rank = new int[size];
		for (int i = 0; i < size; i++) parent[i] = i;
		Arrays.fill(rank, 1);
		unConnectNum = size;
	}

	private int find(int a) {
		if (a > parent.length || a < 0) {
			throw new IllegalArgumentException("FindIndexOutOfBounds:" + a);
		}
		while (parent[a] != a) {
			//路径压缩，两步一跳
			parent[a] = parent[parent[a]];
			a = parent[a];
		}
		return a;
	}

	public boolean isConnect(int a, int b) {
		return find(a) == find(b);
	}

	public int getUnConnectNum() {
		return unConnectNum;
	}

	public int union(int a, int b) {
		//按秩合并，仅当相同时高度+1
		int aroot = find(a);
		int broot = find(b);
		if (aroot == broot) return aroot;
		unConnectNum--;
		if (rank[aroot] > rank[broot]) {
			parent[broot] = aroot;
			return aroot;
		} else if (rank[aroot] < rank[broot]) {
			parent[aroot] = broot;
			return broot;
		} else {
			parent[broot] = aroot;
			rank[aroot]++;
			return aroot;
		}
	}
}
class Game{
    /*
    //n为一次最多走的步数，m为可走的总步数（牌数），go为一次可以走的数量（不一定连续）
    int[] used,sg;
    public int sg(int i){
        //boolean[] used =new boolean[1003];
        for (int j : go) {
            if(j>n||i+j>m)break;
            if(sg[i+j]==-1){
                used[sg(i+j)]=true;
            }else used[m-sg[i+j]]=true;
        }
        for (int k = 0; k <m; k++) {
            if(!used[m-k]) {sg[i]=k;return m-k;}
        }
        return 0;
    }
     */
}
class AdjacencyTable{
	class Node{
		int to,value;
		public Node(int to,int value) {this.to=to;this.value=value;}
	}
	ArrayList<Node>[] map;
	int[] node;
	int size;
	public AdjacencyTable(int i) {
		node = new int[i];
		map=new ArrayList[i];
		size=i;
		for (int i1 = 0; i1 < map.length; i1++) map[i1]= new ArrayList<>();
	}
	public void setRoad(int start, int next, int value){
		map[start].add(new Node(next,value));
	}
}
class GeneratingFunction{
	ArrayList<Double>[] formulaCoefficient;
	int[] addPowerNum,less,more;
	int nowIndex=0,size;
	boolean isTwoPower=false;
	public GeneratingFunction(int size){
		//size：括号数
		//formulaCoefficient =new ArrayList[size];
		this.size=size;
		less=new int[size];
		more=new int[size];
		addPowerNum=new int[size];
	}
	public int setAdd(int a,int b,int c){
		addPowerNum[nowIndex]=a;
		less[nowIndex]=b;
		more[nowIndex]=c;
		return nowIndex++;
	}
	public void setCoeIndex(int index,int value,int max){
		//int num=max/addPowerNum[index];
		ArrayList<Double> now=new ArrayList<Double>(max);
		now.add(0,1.0);
		for (int i = 1; i < max; i++) {
			now.add(i,value*i*1.0);
		}
		formulaCoefficient[index]=now;
	}
	public double[] Generate(int ask){
		double[] a=new double[13],b=new double[13],fact=new double[12];
		for (int i = 0; i < 11; i++) {
			int k=1;
			for (int j = i; j >0 ; j--) {
				k*=j;
			}
			fact[i]=k;
		}
		for (int i = 0; i <= more[0]; i+=addPowerNum[0]) {
			a[i]=1.0/fact[i];
		}//构造第一个括号中多项式
		for (int i = 1; i <= size; i++) {
			if(addPowerNum[i]==0) break;
			//if(addPowerNum[i]>size) break;
            /*for (int j = 0; j <= more[i]; j+=addPowerNum[i]) {
                b[j]=1.0/fact[j];
            }
            double[] temp=new double[13];*/
			for (int j = 0; j <= ask; j++) {
				//if(a[j]!=0)
				for (int k = 0; k+j<=ask&&k <= more[i]; k+=addPowerNum[i]) {
					b[j+k]+=a[j]/fact[k];
				}
			}
			a=Arrays.copyOfRange(b,0,b.length);
			Arrays.fill(b,0);
		}
		for (int i = 0; i <= ask; i++) {
			a[i]*=fact[i];
		}
		return a;
	}

	/**
	 * fft
	 * 未完成，之后待完善，暂时母函数使用n2复杂度实现
	 */



    /*
    完成非递归fft的下标二进制翻转预处理
     */
	public void fftChange(Complex[] a){

	}
	/*
	fft，on为1时是dft，为-1时是idft
	参考：https://blog.csdn.net/qq_37136305/article/details/81184873
	https://www.cnblogs.com/RabbitHu/p/FFT.html
	https://www.jianshu.com/p/a765679b1826
	 */
    /*public Complex[] FFT(Complex[] a,int on){

    }*/
	public void FFTRecursion(Complex[] a, int start, int end, int on){
		int n=end-start+1,m=n/2,mid=(start+end)/2;
		if(n<=1) return;
		Complex[] buffer=new Complex[n];
		for (int i = 0; i < m; i++) {
			buffer[i]=a[start+2*i];
			buffer[i+m]=a[start+2*i+1];
		}
		for (int i = 0; i < n; i++) a[start+i]=buffer[i];
		FFTRecursion(a,start,mid,1);
		FFTRecursion(a,mid+1,end,1);
		for (int i = 0; i < m; i++) {
			//若on为1，x为w（k,n)，实现dft，此处的on无影响
			//若为-1则x为w(-k,n)即为w(k,n)的倒数，后面加一个判断：当为-1时/n，则可实现idft
			Complex x = Complex.omega(n, i * on);
			buffer[i] = a[i].plus(x.multiple(a[i + m]));
			buffer[i + m] = a[i].minus(x.multiple(a[i + m]));
		}
		if(on==-1){
			for (int i = 0; i < buffer.length; i++) {
				buffer[i]=buffer[i].scale(1D/a.length);
			}
		}
		for (int i = 0; i < n; i++) {
			a[start+i]=buffer[i];
		}
	}



	//网上别人的，但是母函数计算结果还是有问题，系数均为同一个值
	public static Complex[] fft(Complex[] x) {
		int N = x.length;

		// base case
		if (N == 1) return new Complex[]{x[0]};

		// radix 2 Cooley-Tukey FFT
		if (N % 2 != 0) {
			throw new RuntimeException("N is not a power of 2");
		}

		// fft of even terms
		Complex[] even = new Complex[N / 2];
		for (int k = 0; k < N / 2; k++) {
			even[k] = x[2 * k];
		}
		Complex[] q = fft(even);

		// fft of odd terms
		Complex[] odd = even;  // reuse the array
		for (int k = 0; k < N / 2; k++) {
			odd[k] = x[2 * k + 1];
		}
		Complex[] r = fft(odd);

		// combine
		Complex[] y = new Complex[N];
		for (int k = 0; k < N / 2; k++) {
			double kth = -2 * k * Math.PI / N;
			Complex wk = new Complex(Math.cos(kth), Math.sin(kth));
			y[k] = q[k].plus(wk.multiple(r[k]));
			y[k + N / 2] = q[k].minus(wk.multiple(r[k]));
		}
		return y;
	}
	public static Complex[] ifft(Complex[] x) {
		int N = x.length;
		Complex[] y = new Complex[N];

		// take conjugate
		for (int i = 0; i < N; i++) {
			y[i] = x[i].conjugate();
		}

		// compute forward FFT
		y = fft(y);

		// take conjugate again
		for (int i = 0; i < N; i++) {
			y[i] = y[i].conjugate();
		}

		// divide by N
		for (int i = 0; i < N; i++) {
			y[i] = y[i].scale(1.0 / N);
		}

		return y;

	}
	public static Complex[] cconvolve(Complex[] x, Complex[] y) {

		// should probably pad x and y with 0s so that they have same length
		// and are powers of 2
		if (x.length != y.length) { throw new RuntimeException("Dimensions don't agree"); }

		int N = x.length;

		// compute FFT of each sequence，求值
		Complex[] a = fft(x);
		Complex[] b = fft(y);

		// point-wise multiply，点值乘法
		Complex[] c = new Complex[N];
		for (int i = 0; i < N; i++) {
			c[i] = a[i].multiple(b[i]);
		}

		// compute inverse FFT，插值
		return ifft(c);
	}
	public static Complex[] convolve(Complex[] x, Complex[] y) {
		Complex ZERO = new Complex(0, 0);

		Complex[] a = new Complex[2*x.length];//2n次数界，高阶系数为0.
		for (int i = 0;        i <   x.length; i++) a[i] = x[i];
		for (int i = x.length; i < 2*x.length; i++) a[i] = ZERO;

		Complex[] b = new Complex[2*y.length];
		for (int i = 0;        i <   y.length; i++) b[i] = y[i];
		for (int i = y.length; i < 2*y.length; i++) b[i] = ZERO;

		return cconvolve(a, b);
	}

}
//向量
class vector{
	//二维向量，叉积为面积
	double x1,y1,x2,y2,x,y;
	public vector(int x1,int y1,int x2,int y2){
		this.x1=x1;
		this.y1=y1;
		this.x2=x2;
		this.y2=y2;
		x=x2-x1;
		y=y2-y1;
	}
	public vector(int x,int y){
		this.x=x;
		this.y=y;
	}
	public double CrossProduct(vector v){
		//x1y2-x2y1
		return x*v.y-y*v.x;
	}

}
//虚数
class Complex {
	//参考：https://www.jianshu.com/p/a765679b1826
	private final double re; // the real part
	private final double im; // the imaginary part

	// create a new object with the given real and imaginary parts
	public Complex(double real, double imag) {
		re = real;
		im = imag;
	}
	//返回单位圆分成n份后，k刻度（第k个）点的虚数系下的坐标，fft用
	public static Complex omega(int n, int k){
		return new Complex(Math.cos(2*Math.PI*k/n),Math.sin(2*Math.PI*k/n));
	}
	// return a string representation of the invoking Complex object
	@Override
	public String toString() {
		if (im == 0)
			return re + "";
		if (re == 0)
			return im + "i";
		if (im < 0)
			return re + " - " + (-im) + "i";
		return re + " + " + im + "i";
	}

	// return abs/modulus/magnitude
	public double abs() {
		return Math.hypot(re, im);
	}

	// return angle/phase/argument, normalized to be between -pi and pi
	public double phase() {
		return Math.atan2(im, re);
	}

	// return a new Complex object whose value is (this + b)
	public Complex plus(Complex b) {
		Complex a = this; // invoking object
		double real = a.re + b.re;
		double imag = a.im + b.im;
		return new Complex(real, imag);
	}

	// return a new Complex object whose value is (this - b)
	public Complex minus(Complex b) {
		Complex a = this;
		double real = a.re - b.re;
		double imag = a.im - b.im;
		return new Complex(real, imag);
	}

	// return a new Complex object whose value is (this * b)
	public Complex multiple(Complex b) {
		Complex a = this;
		double real = a.re * b.re - a.im * b.im;
		double imag = a.re * b.im + a.im * b.re;
		return new Complex(real, imag);
	}

	// scalar multiplication
	// return a new object whose value is (this * alpha)
	public Complex multiple(double alpha) {
		return new Complex(alpha * re, alpha * im);
	}

	// return a new object whose value is (this * alpha)
	public Complex scale(double alpha) {
		return new Complex(alpha * re, alpha * im);
	}

	// return a new Complex object whose value is the conjugate of this
	public Complex conjugate() {
		return new Complex(re, -im);
	}

	// return a new Complex object whose value is the reciprocal of this
	public Complex reciprocal() {
		double scale = re * re + im * im;
		return new Complex(re / scale, -im / scale);
	}

	// return the real or imaginary part
	public double re() {
		return re;
	}

	public double im() {
		return im;
	}

	// return a / b
	public Complex divides(Complex b) {
		Complex a = this;
		return a.multiple(b.reciprocal());
	}

	// return a new Complex object whose value is the Complex exponential of
	// this
	public Complex exp() {
		return new Complex(Math.exp(re) * Math.cos(im), Math.exp(re) * Math.sin(im));
	}

	// return a new Complex object whose value is the Complex sine of this
	public Complex sin() {
		return new Complex(Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im));
	}

	// return a new Complex object whose value is the Complex cosine of this
	public Complex cos() {
		return new Complex(Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im));
	}

	// return a new Complex object whose value is the Complex tangent of this
	public Complex tan() {
		return sin().divides(cos());
	}

	// a static version of plus
	public static Complex plus(Complex a, Complex b) {
		double real = a.re + b.re;
		double imag = a.im + b.im;
		Complex sum = new Complex(real, imag);
		return sum;
	}

	// See Section 3.3.
	@Override
	public boolean equals(Object x) {
		if (x == null)
			return false;
		if (this.getClass() != x.getClass())
			return false;
		Complex that = (Complex) x;
		return (this.re == that.re) && (this.im == that.im);
	}

	// See Section 3.3.
	@Override
	public int hashCode() {
		return Objects.hash(re, im);
	}
}
//分数
class Fraction {

	private long Numerator; // 分子
	private long Denominator; // 分母

	public Fraction(long numerator, long denominator) {
		this.Numerator = numerator;
		if (denominator == 0) {
			throw new ArithmeticException("分母不能为零");
		} else {
			this.Denominator = denominator;
		}
		change();
	}

	public Fraction() {
		this(0, 1);
	}

	public long getNumerator() {
		return Numerator;
	}

	public void setNumerator(long numerator) {
		Numerator = numerator;
	}

	public long getDenominator() {
		return Denominator;
	}

	public void setDenominator(long denominator) {
		Denominator = denominator;
	}

	private Fraction change() {
		long gcd = this.gcd(this.Numerator, this.Denominator);
		this.Numerator /= gcd;
		this.Denominator /= gcd;
		return this;
	}

	/**
	 * 最大公因数
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	private long gcd(long a, long b) {
		long mod = a % b;
		if (mod == 0) {
			return b;
		} else {
			return gcd(b, mod);
		}
	}

	/**
	 * 四则运算
	 * @return
	 */
	public Fraction add(Fraction second) {
		return new Fraction(this.Numerator * second.Denominator + second.Numerator * this.Denominator,
				this.Denominator * second.Denominator);
	}

	public Fraction sub(Fraction second) {
		return new Fraction(this.Numerator * second.Denominator - second.Numerator * this.Denominator,
				this.Denominator * second.Denominator);
	}

	public Fraction multiply(Fraction second) {
		return new Fraction(this.Numerator*second.Numerator,
				this.Denominator * second.Denominator);
	}

	public Fraction devide(Fraction second) {
		return new Fraction(this.Numerator*second.Denominator,
				this.Denominator * second.Numerator);
	}

	@Override
	public String toString() {
		return String.format("{%d/%d}", this.Numerator, this.Denominator);
	}
}