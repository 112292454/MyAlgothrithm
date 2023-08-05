package org.example;



import org.example.whole.Trie;

import java.util.*;

import static java.lang.Math.max;


public class leetcode {

	public boolean isSubtree(TreeNode root, TreeNode subRoot) {
		//val=val,左&右
		//val!=val， 左||右
		return extracted(root, subRoot,subRoot);

	}

	private boolean extracted(TreeNode root, TreeNode subRoot,TreeNode sub) {
		if(root==null^subRoot==null) return false;
		else if(root==null) return true;
		if(root.val!= subRoot.val)
			return extracted(root.left, sub,sub) || extracted(root.right, sub,sub);
		else
			return extracted(root.left, subRoot.left,sub) && extracted(root.right, subRoot.right,sub);
	}


	public LNode connect(LNode root) {
		/*Deque<Node> q=new LinkedList<>();
		if(root!=null) q.addFirst(root);
		while (!q.isEmpty()){
			int s=q.size();
			Node last=null;
			for (int i = 0; i < s; i++) {
				Node now=q.pollLast();
				if(now.right!=null) q.addFirst(now.right);
				if(now.left!=null) q.addFirst(now.left);
				if(last==null) last=now;
				else {
					now.next=last;
					last=now;
				}
			}
		}
		return root;*/
		LNode cur=root;
		while (cur!=null){
			LNode temp=new LNode(),p=temp;
			while (cur!=null){
				if(cur.left!=null){
					p.next=cur.left;
					p=p.next;
				}
				if(cur.right!=null){
					p.next=cur.right;
					p=p.next;
				}
				cur=cur.next;
			}
			cur=temp.next;
		}
		return root;
	}

	private final int[] dx8={0,0,1,1,1,-1,-1,-1};
	private final int[] dy8={1,-1,0,1,-1,0,1,-1};

	private final int[] dx4={0,0,1,-1};
	private final int[] dy4={1,-1,0,0};

	public int shortestPathBinaryMatrix(int[][] grid) {
		Deque<node> q=new LinkedList<>();
		if(grid[0][0]==0) q.addFirst(new node(1,0,0));
		int n=grid.length;
		while (!q.isEmpty()){
			int s=q.size();
			for (int i = 0; i < s; i++) {
				node now=q.pollLast();
				int x=now.x,y=now.y,v=now.value;
				if(x==n-1&&y==n-1) return v;
				grid[x][y]=-1;
				for (int j = 0; j < dx8.length; j++) {
					int nx=x+dx8[j],ny=y+dy8[j];
					if(nx<0||ny<0||nx>=n||ny>=n) continue;
					if (grid[nx][ny] == 0) {
						//grid[nx][ny] = -1;
						q.addFirst(new node(v + 1, nx,ny));
					}
				}
			}
		}
		return -1;
	}

	public void solve(char[][] board) {
		int m=board.length,n=board[0].length;
		char[][] res=new char[m][n];
		for (char[] a : res) {
			Arrays.fill(a,'X');
		}
		for (char[] a : res) {
			System.out.println(Arrays.toString(a));
		}
		for (int i = 0; i < m; i++) {
			if(board[i][0]=='O') solveDFS(board,res,i,0);
			if(board[i][n-1]=='O') solveDFS(board,res,i,n-1);
		}
		for (int i = 0; i < n; i++) {
			if(board[0][i]=='O') solveDFS(board,res,0,i);
			if(board[m-1][i]=='O') solveDFS(board,res,m-1,i);
		}
		for (int i = 0; i < m; i++) {
			board[i]=res[i];
		}
		board=res;
	}
	private void solveDFS(char[][] board,char[][] res,int x,int y){
		//if(board[x][x]=='X') return;
		board[x][y]='X';
		res[x][y]='O';
		for (int i = 0; i < dx4.length; i++) {
			int nx=x+dx4[i],ny=y+dy4[i];
			if(nx<0||ny<0||nx>=board.length||ny>=board[0].length) continue;
			if(board[nx][ny]=='O') {
				res[nx][ny]='O';
				board[nx][nx]='X';
				solveDFS(board, res, nx, ny);
			}
		}
	}

	public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
		List<List<Integer>> res=new ArrayList<>();
		List<Integer> path=new ArrayList<>();
		path.add(0);
		allPathsSourceTargetDFS(graph,res,path);
		return res;
	}
	public void allPathsSourceTargetDFS(int[][] graph,List<List<Integer>> res,List<Integer> path) {
		int now=path.get(path.size()-1);
		if(now==graph.length-1) {
			res.add(new ArrayList<>(path));
			return;
		}
		for (int i : graph[now]) {
			path.add(i);
			allPathsSourceTargetDFS(graph, res, path);
			path.remove(path.size()-1);
		}
	}

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		Arrays.sort(nums);
		return subsetsWithDupDFS(nums,0,new ArrayList<>(),new ArrayList<>(),false);
	}
	public List<List<Integer>> subsetsWithDupDFS(int[] nums, int i, List<List<Integer>> res, List<Integer> now,boolean pre) {
		if(i==nums.length) {
			res.add(new ArrayList<>(now));
			return res;
		}
		subsetsWithDupDFS(nums,i+1,res,now,false);
		if(pre||i <= 0 || nums[i - 1] != nums[i]) {
			now.add(nums[i]);
			subsetsWithDupDFS(nums,i+1,res,now,true);
			now.remove(now.size()-1);
		}
		return res;
	}

	public int rob(int[] nums) {
		//dpi=max(dpi-1,dpi-2+vali)
		int[] dp=new int[nums.length];
		dp[0]=0;dp[1]=nums[1];
		for (int i = 2; i < dp.length; i++) {
			dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i]);
		}
		int res=dp[nums.length-1];
		dp[0]=nums[0];dp[1]=dp[0];
		for (int i = 2; i < dp.length; i++) {
			dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i]);
		}
		res=Math.max(res,dp[nums.length-1]);
		return res;
	}
//55
	public boolean canJump(int[] nums) {
		int max=nums[0];
		for (int i = 1; i < nums.length; i++) {
			if(i>max) return false;
			else if(max>=nums.length) return true;
			max=Math.max(max,i+nums[i]);
		}
		return true;
	}
//45
	public int jump(int[] nums) {
		int max=nums[0],step=1,nextM=0;
		for (int i = 1; i < nums.length; i++) {
			if(i>max) {
				step++;
				max=nextM;
			}
			nextM=Math.max(nextM,i+nums[i]);
			if(nextM>= nums.length) return step;
		}
		return step;
	}
//62
	public int uniquePaths(int m, int n) {
		int[][] map=new int[m][n];
		map[0][0]=1;
		for (int i = 0; i < map.length; i++) {
			for (int j = 0; j < map[i].length; j++) {
				map[i][j]+=i>0?map[i-1][j]:0;
				map[i][j]+=j>0?map[i][j-1]:0;
			}
		}
		return map[m-1][n-1];
	}
//47
	public List<List<Integer>> permuteUnique(int[] nums) {
		Arrays.sort(nums);
		return permuteUniqueDFS(nums,new boolean[nums.length],new ArrayList<>(),new ArrayList<>());
	}
	public List<List<Integer>> permuteUniqueDFS(int[] nums,boolean[] vis,List<Integer> now,List<List<Integer>> total)  {
		if(now.size()==nums.length){
			total.add(new ArrayList<>(now));
			return total;
		}
		for (int i = 0; i < nums.length; i++) {
			if(vis[i]) continue;
			if(i>0&&nums[i]==nums[i-1]&&vis[i-1]) continue;//!!改成！vis会变快

			now.add(nums[i]);
			vis[i]=true;
			permuteUniqueDFS(nums,vis,now,total);
			vis[i]=false;
			now.remove(now.size()-1);
		}
		return total;
	}
//40
	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		Arrays.sort(candidates);
		return combinationSum2DFS(candidates,true,new ArrayList<>(),new ArrayList<>(),target,0,0);
	}
	public List<List<Integer>> combinationSum2DFS(int[] nums,boolean vis,List<Integer> now,List<List<Integer>> total,int tar,int sum,int pos) {
		if (sum == tar) {
			total.add(new ArrayList<>(now));
			return total;
		} else if (sum>tar||pos>=nums.length) {
			return total;
		}

		if (pos > 0 && nums[pos] == nums[pos - 1] && !vis) {//重复的，前一个没选那就这个也不选
			combinationSum2DFS(nums, false, now, total, tar, sum, pos + 1);
		}else {//重复的，前一个选了，就不选和选当前的都试试————这样连续的x中，选中x的数量是单调增的，不会有xx_和x_x的选法共存（只会有第一种：x__,xx_,xxx）
			now.add(nums[pos]);
			combinationSum2DFS(nums, true, now, total, tar, sum+nums[pos], pos + 1);
			now.remove(now.size()-1);
			combinationSum2DFS(nums, false, now, total, tar, sum, pos + 1);
		}
		return total;
	}
//39
	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> total = new ArrayList<>();
		candidates= Arrays.stream(candidates).boxed().sorted((a,b)->b-a).mapToInt(Integer::intValue).toArray();
		combinationSumDFS(candidates, new ArrayList<>(), total, target, 0, 0);
		return total;
	}
	public boolean combinationSumDFS(int[] nums,List<Integer> now,List<List<Integer>> total,int tar,int sum,int pos) {
		//返回值：标识能不能在while里继续加num【pos】：sum>=target时不行
		if(sum>=tar){
			if (sum == tar) total.add(new ArrayList<>(now));
			return false;
		}else if (pos>=nums.length) {
			return  true;
		}

		int cnt=now.size();
		while(combinationSumDFS(nums, now, total, tar, sum, pos+1)){
			now.add(nums[pos]);
			sum+=nums[pos];
		}
		//恢复now在进入函数时的原样——这里很灵性的是，用now=xxx（比如之前存的clone值)是不行的
		//因为上一层函数里的now指向堆里某个区域a；此函数内clone了一个list指向堆上的区域b
		// 此时写now=clone，只会让本函数里的now指向b，不会把上一级里的now指针由a变成b，所以上一层依然是被修改的，没有复原
		while(now.size()>cnt){
			now.remove(now.size()-1);
		}
		return true;
	}
//22
	public List<String> generateParenthesis(int n) {
		return generateParenthesisDFS(0,0,n,0,new ArrayList<>(),new char[n+n]);
	}
	public List<String> generateParenthesisDFS(int bias,int end,int max,int depth,List<String> res,char[] temp) {
		if(depth==temp.length) {
			res.add(new String(temp));
			return res;
		}
		if(bias<max){
			temp[depth]='(';
			generateParenthesisDFS(bias+1, end, max, depth+1, res, temp);
		}
		if(bias>end){
			temp[depth]=')';
			generateParenthesisDFS(bias, end+1, max, depth+1, res, temp);
		}
		return res;
	}
//79
	public boolean exist(char[][] board, String word) {
		boolean[][] vis = new boolean[board.length][board[0].length];
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if(board[i][j]==word.charAt(0)&&existDFS(board, vis,word,1,i,j)) return true;
			}
		}
		return false;
	}
	public boolean existDFS(char[][] board,boolean[][] vis, String word,int index,int x,int y) {
		for (int i = 0; i < 4; i++) {
			int nx=dx4[i]+x,ny=dy4[i]+y;
			if(nx<0||ny<0||nx>=board.length||ny>=board[0].length||vis[nx][ny]) continue;
			vis[nx][ny]=true;
			if(board[nx][ny]==word.charAt(index)&&(index==word.length()-1||existDFS(board,vis, word, index+1, nx, ny))) return true;
			vis[nx][ny]=false;
		}
		return false;
	}

	public List<String> removeSubfolders(String[] folder) {
		long time=System.currentTimeMillis();
		for (int i = 0; i < folder.length; i++) {
			folder[i]=folder[i].replace('/', (char) ('z'+1))+'{';
		}
		Trie trie=new Trie(27,'a');
		System.out.println(System.currentTimeMillis() - time);
		for (String s : folder) {
			trie.add(s);
		}
		System.out.println(System.currentTimeMillis() - time);

		List<String> allWord = trie.listAllWord();
		System.out.println(System.currentTimeMillis() - time);

		List<String> res=new ArrayList<>();
		for (String s : allWord) {
			res.add(s.replace('{', '/').substring(0,s.length()-1));
		}
		System.out.println(System.currentTimeMillis() - time);

		return res;
	}
//583
public int minDistance(String word1, String word2) {
	String temp= word1.length() >= word2.length() ? word1 : word2;
	word1=word1.length()<word2.length()?word1:word2;
	word2=temp;//1m短，2n长
	int m=word1.length(),n=word2.length();
	int[][] dp=new int[n][m];
	for (int[] ints : dp) Arrays.fill(ints, m+n);
	dp[0][0]=word2.charAt(0)==word1.charAt(0)?0:2;
	for (int i = 1; i < n; i++) {
		if(word1.charAt(0)==word2.charAt(i)) dp[i][0]=i;
		else dp[i][0]=Math.min(i+2, dp[i-1][0]+1);
	}
	for (int i = 1; i < m; i++) {
		if(word1.charAt(i)==word2.charAt(0)) dp[0][i]=i;
		else dp[0][i]=Math.min(i+2, dp[0][i-1]+1);
	}


	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			if(word1.charAt(i)==word2.charAt(j))
				dp[j][i]=Math.min(dp[j-1][i-1],i+j);
			else
				dp[j][i]=Math.min(dp[j][i-1], dp[j-1][i])+1;
		}
	}
	return dp[n-1][m-1];
}
	public boolean isReverse(int a) {
		int[] num=new int[10];
		int i=0,t=0;
		while (a>0) {
			num[i++]=a%10;
			a/=10;
		}
		for (int j = 0; j <= i; j++) {
			t=t*10+num[i--];
		}
		return t==a;
	}

	//1590
	public int minSubarray(int[] nums, int p) {
		long[] sum=new long[nums.length];
		sum[0]=nums[0];
		HashMap<Long, Integer> index=new HashMap<>();
		index.put(sum[0]%p,0);
		for (int i = 1; i < nums.length; i++) {
			sum[i]=sum[i-1]+nums[i];
			index.put(sum[i]%p,i);
		}
		long total=sum[sum.length - 1],sur=total%p;
		if(sur==0) return 0;
		int res=1<<31-1;
		for (int i = sum.length-1; i >=0 ; i--) {
			long a=(sum[i]%p+p-sur)%p,b=a+sur+sur;
			if(index.containsKey(a)&&sum[i]>p) {
				res=Math.min(res,Math.abs(i-index.get(a))+1);
			}
			//if(index.containsKey(b)) res=Math.min(res,i-index.get(b));
			//System.out.println(res);
		}
		//System.out.println("\n");
		return res==1<<31-1?-1:res==0?1:res;
	}

	//17.05
	public String[] findLongestSubarray(String[] array) {
		HashMap<Integer, Integer> h=new HashMap<>();
		h.put(0,0);
		int sum=0;
		int l=0,r=-1;
		for (int j=0; j < array.length;j++) {

			char a =array[j].charAt(0);
			int b = (a-'0'>=0&&a-'0'<=9)?1:-1;
			sum=sum+b;
			if(h.containsKey(sum)){
				int ll=h.get(sum);
				if(j-ll>r-l) {
					r=j;
					l=ll;
				}
			}else{
				h.put(sum,j+1);
			}
		}
		if(sum==0) {
			r=array.length-1;
			l=0;
		}
		if(r>l) return Arrays.copyOfRange(array, l,r+1);
		else return new String[0];
	}

	//2383
	public int minNumberOfHours(int initialEnergy, int initialExperience, int[] energy, int[] experience) {
		int e=0,exp=0;
		for (int i = energy.length - 1; i >= 0; i--) {
			e+=energy[i];
			exp+=experience[i]-energy[i];
		}
		return Math.max(e+exp-initialEnergy-initialExperience,0);
	}

	//1605
	public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
		int[][] mat=new int[rowSum.length][colSum.length];
		mat[0]=colSum;
		for (int i = 0; i < mat.length-1; i++) {
			int sum = 0;
			for (int j = 0; j < mat[i].length; j++) {
				if(sum<rowSum[i]){
					int need=rowSum[i]-sum;
					if(need>mat[i][j]){
						sum+=mat[i][j];
					}else{
						sum=rowSum[i];
						mat[i+1][j]+=mat[i][j]-need;
						mat[i][j]=need;
					}
				}else{
					mat[i+1][j]+=mat[i][j];
					mat[i][j]=0;
				}
			}
		}
		return mat;
	}

	//2389
	public int[] answerQueries(int[] nums, int[] queries) {
		Arrays.sort(nums);
		int[] sum=new int[nums.length];
		sum[0]=nums[0];
		for (int i = 1; i < nums.length; i++) {
			sum[i]=sum[i-1]+nums[i];
		}
		int[] res=new int[queries.length];
		for (int i = 0; i < queries.length; i++) {
			int k = answerQueriesBI(sum, queries[i]);
			if(sum[i]==queries[i]) k++;
			res[i]=k;
		}
		return res;
	}
	public int answerQueriesBI(int[] nums, int tar) {
		int l=0,r=nums.length-1,mid=0;
		while (l<=r){
			mid=(l+r)/2;
			if(nums[mid]==tar) return mid;
			else if(nums[mid]>tar) r=mid-1;
			else l=mid+1;
		}
		return l;
	}

	//1616
	public boolean checkPalindromeFormation(String a, String b) {
		char[] aa=a.toCharArray();
		char[] bb=b.toCharArray();
		int n=aa.length;
		for (int i = 0; i < n; i++) {
			//if(n%2==1&&Math.abs(i-(n-i))<=1||i==n-i) return true;
			if(aa[i]!=bb[n-i-1]) {
				if(check(aa,i,n-i-1)||check(bb,i,n-i-1)) return true;
				break;
			}else if(i>=n-i) return true;
		}
		for (int i = 0; i < n; i++) {
			//if(n%2==1&&Math.abs(i-(n-i))<=1||i==n-i) return true;
			if(bb[i]!=aa[n-i-1]) {
				if(check(aa,i,n-i-1)||check(bb,i,n-i-1)) return true;
				break;
			}else if(i>=n-i) return true;
		}
		return check(aa,0,n - 1)||check(bb,0, n-1);
	}
	private boolean check(char[] a,int l,int r){
		if(l>=r) return true;
		for (; l <= r; l++,r--) {
			if(a[l]!=a[r]) return false;
		}
		return true;
	}

	public int diffFrequencyStr(String word) {
		HashMap<Character,Integer> hash=new HashMap<>();
		for (int i = 0; i < word.length(); i++) {
			hash.put(word.charAt(i), hash.getOrDefault(word.charAt(i),0)+1);
		}
		int[] times=new int[26];
		hash.forEach((k,v)->times[k-'a']=v);
		Arrays.sort(times);
		int res=0;
		for (int i = 0; i < times.length-1; i++) {
			if(times[i+1]==i) res++;
		}
		return res;
	}

	//Off06
	public int[] reversePrint(ListNode head) {
		int len=0;
		ListNode t=head;
		while (t!=null){
			len++;
			t=t.next;
		}
		int[] res=new int[len--];
		for(;len>=0;len--){
			res[len]=head.val;
			head=head.next;
		}
		return res;
	}


	//Off57
	public int[][] findContinuousSequence(int T) {
		int maxPos= (int) Math.sqrt(T*2)+1;
		List<int[]> temp=new ArrayList<>();
		for (int a = 1; a <= T/2; a++) {
			long c= (long) (a - T) <<1;
			long b=a+a+1;
			double delta=Math.sqrt(b*b-4*c);
			if(Math.round(delta)!=delta) continue;
			int l= (int) ((delta-b)/2);
			if((l*l+b*l+c==0)){
				temp.add(new int[]{a,l});
			}else{
				l= (int) ((delta+b)/2);
				temp.add(new int[]{a,l});
			}
		}
		int[][] res=new int[temp.size()][];
		for (int i = 0; i < res.length; i++) {
			int start=temp.get(i)[0],len=temp.get(i)[1];
			res[i]=new int[len+1];
			for (int j = 0; j <= len; j++) {
				res[i][j]=start+j;
			}
		}
		return res;
	}

	//Off62
	public int lastRemaining(int n, int m) {
		int f = 0;
		for (int i = 2; i != n + 1; ++i) {
			f = (m + f) % i;
		}
		return f;
	}

	//Off24
	public ListNode reverseList(ListNode head) {
		ListNode cur=head,next=cur.next;
		head.next=null;
		while (next!=null){
			ListNode temp=next.next;
			next.next=cur;
			cur=next;
			next=temp;
		}
		return cur;
	}

	//Off35
//	public Node copyRandomList(Node head) {
//		if(head==null) return null;
//		HashMap<Node,Integer> nodeIndex=new HashMap<>();
//		List<Node> nodes=new ArrayList<>();
//		while (head!=null){
//			nodes.add(head);
//			nodeIndex.put(head,nodes.size() - 1);
//			head=head.next;
//		}
//		Node[] res=new Node[nodes.size()];
//		for (int i = 0; i < res.length; i++) {
//			res[i]=new Node(0);
//		}
//		for (int i = 0; i < res.length; i++) {
//			Node n = nodes.get(i);
//			res[i].val=n.val;
//			if(i>0) res[i-1].next=res[i];
//			Object Rindex=nodeIndex.getOrDefault(n.random,null);
//			if(Rindex!=null) res[i].random=res[(int)Rindex];
//			else res[i].random= (Node) Rindex;
//		}
//		return res[0];
//	}

	//Off04
	public boolean findNumberIn2DArray(int[][] matrix, int target) {
		int x=0,y=matrix[0].length-1;
		while (checkInBounds(matrix,x,y)){
			if(matrix[x][y]==target) return true;
			else if(matrix[x][y]>target) y--;
			else x++;
		}
		return false;
	}
	private boolean checkInBounds(int[][] m,int x,int y){
		return x>=0&&y>=0&&x<m.length&&y<m[0].length;
	}

	//Off50
	public char firstUniqChar(String s) {
		int[] left=new int[26];
		Arrays.fill(left, -1);
		char[] chars=s.toCharArray();
		int k=0;
		for (char t : chars) {
			t-='a';
			if(left[t]==-1) left[t]=k;
			else left[t]=0x3f3f3f3f;
			k++;
		}
		Arrays.sort(left);
		for (int i : left) {
			if(i!=-1&&i!=0x3f3f3f3f) return s.charAt(i);
		}
		return ' ';
	}

	//Off48


	//32
	public int longestValidParentheses(String s) {
		/**
		 dp[i]为以i结尾的子串的最大长度
		 若dp【i-1】合法，则dp【i】=dp【i-1 -dp【i-1】】（合法段之前的一个字符和i这个字符是不是刚好（），如果不是，就为0，如果是，就为dp【i-1】+2）
		 若i-1不合法，查看i结尾是否有合法的
		 i之前，最后一个合法段，前面的和i这两个字符是否匹配，是
		 */
		char[] ch=s.toCharArray();
		int[] dp=new int[ch.length];
		if(s.length()<2) return 0;
		for (int i = 1; i < dp.length; i++) {
			if(ch[i]=='(') continue;
			if(ch[i-1]=='(') dp[i]=(i>1?dp[i-2]:0)+2;
			else if(dp[i-1]!=0){
				if( (i-1 - dp[i-1]>=0) && (ch[i-1 - dp[i-1]]=='(') ) {
					dp[i]=dp[i-1]+2;
					if(i-2 - dp[i-1]>0) dp[i]+=dp[i-2 - dp[i-1]];
				}
			}
		}
		int res=0;
		for (int i : dp) {
			res=Math.max(res,i);
		}
		return res;
	}

	//95
	public List<TreeNode> generateTrees(int n) {
		return generateTreesBuild(new List[n+1][n+1],1, n);
	}

	private List<TreeNode> generateTreesBuild(List<TreeNode>[][] used,int l,int r){
		if(l>=r){
			List<TreeNode> res=new ArrayList<>();
			if(l==r) res.add(new TreeNode(l));
			else res.add(null);
			return res;
		}else if(used[l][r]!=null) return used[l][r];

		List<TreeNode> res=new ArrayList<>();
		for (int i = l; i <= r; i++) {
			List<TreeNode> rSubTree = generateTreesBuild(used,i+1, r);
			List<TreeNode> lSubTree = generateTreesBuild(used,l, i-1);
			for (TreeNode treeNode : lSubTree) {
				for (TreeNode node : rSubTree) {
					TreeNode root = new TreeNode(i);
					root.left = treeNode;
					root.right = node;
					res.add(root);
				}
			}
		}
		used[l][r]=res;
		return res;
	}
	public int lengthOfLongestSubstring(String s) {
		char[] chars = s.toCharArray();
		int[] leftIndex=new int[26];
		int l=0,r=0,res=-1;
		for (int i = 0; i < chars.length; i++) {
			int c = chars[i] - '0';
			r=i;
			if(leftIndex[c]!=0) {
				l=leftIndex[c];
				res=Math.max(res,r-l);
			}else{
				leftIndex[c]=i;
			}
		}
		return res;
	}

	//1090
	public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
		node[] arrs=new node[values.length];
		for (int i = 0; i < values.length; i++) {
			arrs[i]=new node(0, values[i],labels[i]);
		}
		Arrays.sort(arrs,Comparator.comparingInt(a->a.x));
		Map<Integer,Integer> cnts=new HashMap<>();
		int res=0;
		for (int i = arrs.length - 1; i >= 0&&numWanted>=0; i--) {
			if(!cnts.containsKey(arrs[i].y)||cnts.get(arrs[i].y)<useLimit){
				res+=arrs[i].x;
				cnts.put(arrs[i].y,cnts.getOrDefault(arrs[i].y,0)+1);
				numWanted--;
			}
		}
		return res;
	}

	//84
	public int largestRectangleArea(int[] heights) {
		Deque<Integer> s=new LinkedList<>();
		s.addFirst(1<<31);
		int max=1<<31;
		for (int i = 0; i < heights.length; i++) {
			while (!s.isEmpty()&&s.peek()>=0&&heights[s.peek()]>=heights[i]){
				int out=s.poll();
				if(s.peek()>0) max=Math.max(max,heights[out]*s.peek());
				else max=Math.max(max,heights[out]*i);
			}
			s.push(i);
		}

		while (s.peek()!=1<<31){
			int out=s.poll();
			if(s.peek()==1<<31) max=Math.max(max,heights[out]*out);
			else max=Math.max(max,heights[s.peek()]*(out-s.peek()));
		}
		return max;
	}

	//1186
	public int maximumSum(int[] arr) {
		int[][] dp=new int[arr.length][2];
		int n=arr.length-1,ret=1<<31;
		for(int ele:arr) ret=Math.max(ret,ele);
		dp[0][0]=arr[0];
		dp[0][1]=arr[0];
		//dp.i.0=dp.i-1.0+arr.i/arr.i
		//dp.i.1=dp.i-1.0/ap.i-1.1+arr.i
		for (int i = 1; i < arr.length; i++) {
			dp[i][0]=Math.max(arr[i],dp[i-1][0]+arr[i]);
			dp[i][1]=Math.max(dp[i-1][0],dp[i-1][1]+arr[i]);
			ret=Math.max(ret,Math.max(dp[i][0], dp[i][1]));
		}
		return ret;
	}

	/**
	 * 升序
	 * @param arr
	 * @param target
	 * @return
	 */
	public int binarySearch(int[] arr,int target){
		int l=0,r=arr.length-1,mid=0;
		while (l<r){
			mid=(l+r)/2;
			if(arr[mid]==target) return mid;
			else if (arr[mid]<target) l=mid+1;
			else r=mid-1;
		}
		return l;
	}


	//6899
	public int maximumJumps(int[] nums, int target) {
		int[] dp=new int[nums.length];
		dp[dp.length-1]=-1;

		for (int i = 0; i < nums.length; i++) {
			for (int j = 0; j < i; j++) {
				if(Math.abs(nums[i]-nums[j])<=target
				&&(j==0||dp[j]>0)){
					dp[i]=Math.max(dp[i],dp[j]+1);
				}
			}
		}
		return dp[dp.length-1];
	}
	//6912
	public int maxNonDecreasingLength(int[] nums1, int[] nums2) {
		int n=nums1.length;
		int[][] dp=new int[n][2];
		dp[0][0]=dp[0][1]=1;
		for (int i = 1; i < n; i++) {
			dp[i][0]=maxLenClac(i-1,nums1,nums2,dp,nums1[i]);
			dp[i][1]=maxLenClac(i-1,nums1,nums2,dp,nums2[i]);
		}
		return Arrays.stream(Arrays.stream(dp).max(Comparator.comparingInt(a->Math.max(a[0],a[1]))).get()).max().getAsInt();
	}
	private int maxLenClac(int last,int[] from0,int[] from1,int[][] len,int now){
		return Math.max(now>=from0[last]?len[last][0]+1:1,
				now>=from1[last]?len[last][1]+1:1);
	}

	public boolean isGood(int[] nums) {
		int[] t=new int[nums.length-1];
		for (int num : nums) {
			if(num>nums.length-1) return false;

			t[num-1]++;
		}
		for (int i = 0; i < t.length; i++) {
			if(t[i]!=1&&i!=nums.length-2) return false;
			if(t[i]!=2&&i==nums.length-2) return false;
		}
		return true;
	}

	private boolean ok(char v){
		return v == 'a' || v == 'e' || v == 'i' || v == 'o' || v == 'u' || v == 'A' || v == 'E' || v == 'I' || v == 'O' || v == 'U';
	}
	public String sortVowels(String s) {


		int n = s.length();
		List<Character> cs=new ArrayList();
		char[] chars = s.toCharArray();
		for (int i = 0; i < chars.length; i++) {
			if(ok(chars[i])) cs.add(chars[i]);
		}
		cs.sort(Character::compareTo);
		List<Character> res=new ArrayList();

		int j = 0;
		for (int i= 0; i < n; i++ ){
			if (!ok(chars[i]) ){
				res.add(chars[i]);
			} else {
				res.add(cs.get(j));
				j++;
			}
		}
		StringBuilder sb=new StringBuilder();
		res.forEach(a->sb.append(a));

		return sb.toString();
	}

	public long maxScore(int[] nums, int x) {
		int len=nums.length;
		long[] dp=new long[len];
		int omax=0,emax=0;
		if(nums[0]%2==1) emax=-1;
		else omax=-1;

		dp[0]=nums[0];
		for (int i = 1; i < nums.length; i++) {
			boolean isOdd=nums[i]%2==1;

			long oa=omax==-1?Long.MIN_VALUE:dp[omax]+(isOdd?nums[i]:nums[i]-x);
			long ea=emax==-1?Long.MIN_VALUE:dp[emax]+(isOdd?nums[i]-x:nums[i]);
			if(isOdd){
				if(omax==-1||oa>dp[omax]) omax=i;
			}else {
				if(emax==-1||ea>dp[emax]) emax=i;
			}

			dp[i]=Math.max(oa,ea);
		}
		return Arrays.stream(dp).max().getAsLong();
	}

	class Pack {
		//含01、完全、多重背包，混合背包可并入多重,二维待定
		int[] weight, value, multiple;
		int nums;

		public Pack(int nums) {
			this.nums = nums;
			weight = new int[nums];
			value = new int[nums];
		}

		public long ZeroOnePackCount(int v) {
			//得到总价值为value时选择的方案数，容量不做限制
			//dp【j】【i】为考虑前j个物品，价值为i时的方案数，j维度通过滚动for删除
			//dp[i]=i<value[j]?  dp[i] : dp[i]+dp[i-value[j]]
			//与上面相同，应倒序，保证dpi是从没有选择过物品j的状态转移而来
			long[] dp = new long[v + 1];
			dp[0] = 1;
			for (int j = 0; j < nums; j++) {
				for (int i = v; i >= 0; i--) {
					dp[i] = i < value[j] ? dp[i] : dp[i] + dp[i - value[j]];
					dp[i] %= (1e9 + 7);
				}
			}
			return dp[v];
		}
	}

	public int numberOfWays(int n, int x) {
		int[] dp=new int[n+1];
		int max= (int) Math.pow(n,1.0/x)+2;
		int[] nums=new int[max];
		for (int i = 1; i <= max; i++) {
				nums[i-1]= (int) Math.pow(i,x);
				//value
		}

		Pack p=new Pack(max);
		p.value=nums;
		return (int) p.ZeroOnePackCount(n);

//		dp[0]=1;
//		for (int i = 0; i < dp.length; i++) {
//			for (int j = 0; j < nums.length; j++) {
//				int pow=nums[j];
//				if(i-pow>=0){
//					dp[i]+=dp[i-pow];
//				}
//			}
//		}

	}


	  public static void main(String[] args) {

		  leetcode l = new leetcode();
		  l.maxScore(new int[]{9,58,17,54,91,90,32,6,13,67,24,80,8,56,29,66,
						  85,38,45,13,20,73,16,98,28,56,23,2,47,85,11,97,72,
				  2,28,52,33},90);
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
//Off09
class CQueue {
	//两个栈实现队列

	Deque<Integer> s1,s2;
	public CQueue() {
		s1=new ArrayDeque<>();
		s2=new ArrayDeque<>();
	}

	//public void appendTail(int value) {
	//	if(s1.isEmpty()){
	//		while (!s2.isEmpty()) s1.push(s2.pop());
	//		s1.push(value);
	//	}else{
	//		while (!s1.isEmpty()) s2.push(s1.pop());
	//		s2.push(value);
	//	}
	//}
//
	//public int deleteHead() {
	//	return s1.isEmpty()?s2.pop():s1.pop();
	//}


	public void appendTail(int value) {
		s2.push(value);
	}

	public int deleteHead() {
		if(s1.isEmpty()&&s2.isEmpty()) return -1;
		else if(s1.isEmpty()) while (!s2.isEmpty()) s1.push(s2.pop());
		return s1.pop();
	}
}

//Off30
class MinStack {

	Deque<Integer> s,ms;
	/** initialize your data structure here. */
	public MinStack() {
		s=new LinkedList<>();
		ms=new LinkedList<>();
		ms.push(Integer.MAX_VALUE);
	}

	public void push(int x) {
		s.push(x);
		ms.push(Math.min(ms.peek(),x));
	}

	public void pop() {
		s.pop();
		ms.pop();
	}

	public int top() {
		return s.peek();
	}

	public int min() {
		return ms.peek();
	}
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class LNode {
	public int val;
	public LNode left;
	public LNode right;
	public LNode next;

	public LNode() {}

	public LNode(int _val) {
		val = _val;
	}

	public LNode(int _val, LNode _left, LNode _right, LNode _next) {
		val = _val;
		left = _left;
		right = _right;
		next = _next;
	}
}

class ListNode {
	int val;
	ListNode next;
	ListNode(int x) { val = x; }
}

class Node {
	int val;
	Node next;
	Node random;

	public Node(int val) {
		this.val = val;
		this.next = null;
		this.random = null;
	}
}