package org.example;


import java.util.*;


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

	public static void main(String[] args) {
		leetcode l=new leetcode();
		l.minDistance("algorithm","altruistic");
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