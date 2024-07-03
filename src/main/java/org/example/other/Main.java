package org.example.other;

import java.io.*;
import java.math.BigInteger;
import java.time.LocalDate;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static java.lang.Math.max;
// 1:无需package
// 2: 类名必须Main, 不可修改

public class Main {
    public static void main(String[] args) {
        Pack pack = new Pack(new int[]{71, 69, 1}, new int[]{100, 1, 2});
        System.out.println(pack.ZeroOnePack(70));
        Scanner scan = new Scanner(System.in);
//        solve3C(scan);
        scan.close();
    }

    private static void solve3A(Scanner in) throws IOException {
        DataInputStream reader = new DataInputStream( new FileInputStream("dict.dic"));
        int n=in.nextInt();
        for (int i = 1; i <= n; i++) {
            int a ;
            boolean b ;
            double c ;
            String d ;
            a=reader.readInt();
            b=reader.readBoolean();
            c=reader.readDouble();
            d=reader.readUTF();

            if(i==n){
                System.out.println(a + "\n" + b + "\n" + c + "\n" + d + "\n");
            }
        }
    }
    private static void solve3B(Scanner in) throws IOException, ClassNotFoundException {
        ObjectInputStream reader = new ObjectInputStream( new FileInputStream("dict.dic"));
        int n=in.nextInt();
        for (int i = 1; i <= n; i++) {
            Book b= (Book) reader.readObject();
            if(i==n){
                System.out.println(b.toString());
            }
        }
    }
    private static void solve3C(Scanner in){
        String type=in.nextLine();
        int n=in.nextInt();
        in.nextLine();
        StringBuilder sb=new StringBuilder();
        for (int i = 0; i < n; i++) {
            Double[] vars= Arrays.stream(in.nextLine().trim().split(" "))
                    .map(Double::parseDouble).toArray(Double[]::new);
            myComplex a=new myComplex(vars[0], vars[1]),b=new myComplex(vars[2], vars[3]);
            switch (type) {
                case "add"-> sb.append(a.plus(b).toString());
                case "sub"-> sb.append(a.minus(b).toString());
                case "mul"-> sb.append(a.multiple(b).toString());
                case "div"-> {
                    if(b.re()==0&&b.im()==0){
                        sb.append("Error No : 1001\nError Message : Divide by zero.");
                    }else{
                        sb.append(a.divides(b).toString());
                    }
                }
            }
            sb.append("\n");
        }
        System.out.println(sb);
    }
    private static void solve3D(Scanner in){
        int n=in.nextInt();
        List<Integer> p=new ArrayList<>();
        for (int i = 0; i < n; i++) {
            p.add(in.nextInt());
        }
        p.sort(Integer::compareTo);
        int res=0;
        for (int i = 0; i < (n + 1) / 2; i++) {
            res+=(p.get(i)+1)/2;
        }
        System.out.println(res);
    }
    private static void solve3E(Scanner in){
        int n=in.nextInt();
        Deque<Integer> q=new LinkedList<>();
        for (int i = 0; i < n; i++) {
            q.addFirst(in.nextInt());
        }
        in.nextLine();

        List<Integer> outI=new ArrayList<>();
        while (in.hasNext()){
            String[] args=in.nextLine().trim().split(" ");
            if(args[0].equals("out")){
                int times=Integer.parseInt(args[1]);
                for (int j = 0; j < times&&!q.isEmpty(); j++) {
                    outI.add(q.pollLast());
                }
            }else{
                for (int j = 1; j < args.length; j++) {
                    q.addFirst(Integer.parseInt(args[j]));
                }
            }
        }

        StringBuilder outs=new StringBuilder();
        StringBuilder outa=new StringBuilder();

        if(outI.isEmpty()){
            outs.append("len = 0 ");
        }else{
            outs.append("len = ").append(outI.size()).append(", data = ");
            for (Integer integer : outI) {
                outs.append(integer).append(" ");
            }
            outs.deleteCharAt(outs.length()-1);
        }

        if(q.isEmpty()){
            outa.append("len = 0 ");
        }else{
            outa.append("len = ").append(q.size()).append(", data = ");
            while (!q.isEmpty()){
                outa.append(q.pollLast()).append(" ");
            }
            outa.deleteCharAt(outa.length()-1);
        }

        System.out.println(outa);
        System.out.println(outs);

    }

    private static void solve2A(Scanner in){
        int n=in.nextInt();
        in.nextLine();
        Map<String,String> hash=new HashMap<>();
        for (int i = 0; i < n; i++) {
            String[] strs=in.nextLine().split(" ");
            hash.put(strs[1].trim(),strs[0].trim());
        }
        while (in.hasNext()){
            String s=in.nextLine();
            if(s.equals("dog")) break;
            if (hash.containsKey(s)) System.out.println(hash.get(s));
            else {
                System.out.println("dog");
            }
        }
    }
    private static void solve2B(Scanner in){
        int n=in.nextInt();
        List<Integer> al=new ArrayList<>();
        for (int i = 0; i < n; i++) {
            al.add(in.nextInt());
        }
        int m=in.nextInt();
        List<Integer> bl=new ArrayList<>();
        for (int i = 0; i < m; i++) {
            bl.add(in.nextInt());
        }
        List<Integer> l2=new ArrayList(al);
        l2.addAll(bl);
        l2=l2.stream().distinct().sorted().collect(Collectors.toList());
        List<Integer> l1=al.stream().filter(bl::contains).sorted().collect(Collectors.toList());
        List<Integer> l3=al.stream().filter(a->!l1.contains(a)).sorted().collect(Collectors.toList());
        for (Integer integer : l1) {
            System.out.print(integer + " ");
        }
        System.out.println();
        for (Integer integer : l2) {
            System.out.print(integer + " ");
        }
        System.out.println();
        for (Integer integer : l3) {
            System.out.print(integer + " ");
        }
    }
    private static void solve2C(Scanner in){
        Card first = new Card(in.nextLine());
        Card sec = new Card(in.nextLine());
        Handle hand=new Handle();
        hand.add(first);
        hand.add(sec);
        StringBuilder sb=new StringBuilder();
        while (in.hasNext()) {
            if (hand.getValue() < 17) {
                sb.append("Hit\n");
                Card card = new Card(in.nextLine());
                hand.add(card);
                sb.append(card.type).append(" ");
                if(!card.value.equals("A")) sb.append(card.getComV()).append("\n");
                else sb.append("1 11\n");
            }

            if (hand.getValue() >= 17) {
                break;
            }
        }

        sb.append("Stand\n").append(hand.toString()).append("\n");
        if (hand.getValue() == 21&&hand.size()==2) {
            sb.append("Blackjack");
        } else if (hand.getValue() > 21) {
            sb.append("Bust");
        } else {
            sb.append(hand.getValue());
        }
        System.out.println(sb);
    }
    private static void solve2D(Scanner in){
        ArrayList<Employee> employees=new ArrayList<>();

        int n=in.nextInt();
        for (int i = 0; i < n; i++) {
            int type=in.nextInt();
            String[] params=in.nextLine().trim().split(" ");
            Employee e=null;
            switch (type) {
                case 0 -> e = new SalaridEmployee(params);
                case 1 -> e = new HourlyEmployee(params);
                case 2 -> e = new CommisionEmployee(params);
                case 3 -> e = new basePlusCommisionEmployee(params);
            }
            assert e!=null;
            employees.add(e);
        }
        employees.sort(Employee::compareTo);
        int m=in.nextInt();
        StringBuilder sb=new StringBuilder();
        for (int j = 0; j < m; j++) {
            int type=in.nextInt();
            String param=in.nextLine().trim();
            List<Employee> res=new ArrayList<>();
            switch (type) {
                case 1-> employees.forEach(a->{
                    if(a.socialSecurityNumber.equals(param)) res.add(a);
                });
                case 0-> employees.forEach(a->{
                    if(a.firstName.equals(param)) res.add(a);
                });
            }
            res.forEach(a->sb.append(a.toString()).append("\n"));
        }
        System.out.println(sb);
    }
    private static void solve2E(Scanner in){
        String p=in.nextLine().replace("*", ".*"),lessP=p.replace(".*", ".*?");
        String tar=in.nextLine();

        Pattern a=Pattern.compile(lessP);
        Matcher m1 = a.matcher(tar);
        if(m1.find()) System.out.println(m1.group());

        Pattern b=Pattern.compile(p);
        Matcher m2 = b.matcher(tar);
        if(m2.find()) System.out.println(m2.group());

    }

    private static void solve2211(Scanner scan){
        int n=scan.nextInt();
        BigInteger res=BigInteger.valueOf(0);
        BigInteger a=BigInteger.valueOf(0);
        for (int i = 0; i < n; i++) {
            a=a.multiply(BigInteger.valueOf(10)).add(BigInteger.valueOf(6));
            res=res.add(a.pow(2));
        }
        System.out.println(res);
    }
    private static void solve2204(Scanner scan){
        HashMap<String,Integer> index=new HashMap<>();
        int a=1,b=1;
        for (int i = 3; i < 200; i++) {
            System.out.print(a+"  ");
            int c=a+b;
            a=b;
            b=c;
            a%=10;
            b%=10;
            String s=a+""+b;
            if(index.containsKey(s)){
                System.out.println("\n\n"+s+":"+index.get(s));
                break;
            }else index.put(s,i);
        }
    }
    private static void solve2230(Scanner scan){
        long n=scan.nextInt(),sum=n,res=100000;
        long start=System.currentTimeMillis();
        while (System.currentTimeMillis()-start<2500&&sum<Long.MAX_VALUE-n){
            res=Math.min(res, getCnt(sum));
            sum+=n;
        }
        System.out.println(res);
    }
    private static int getCnt(long n) {
        int cnt= n ==0?0:1;
        while ((n &=(n -1))!=0) cnt++;
        return cnt;
    }
}

class Card implements Comparable{
    String value;
    String type;
    public static Map<String,Integer> sortMap=new HashMap<>();
    public static Map<String,Integer> tempV=new HashMap<>();

    static {
        sortMap.put("A", 1);
        for (int i = 2; i <= 10; i++) {
            sortMap.put((i+""), i);
        }
        sortMap.put("J",10);
        sortMap.put("Q",10);
        sortMap.put("K",10);

        sortMap.put("Spade",100);
        sortMap.put("Heart",101);
        sortMap.put("Diamond",102);
        sortMap.put("Club",103);

        tempV.put("10", 10);
        tempV.put("J", 11);
        tempV.put("Q", 12);
        tempV.put("K", 13);
    }

    public Card(String all) {
        String[] strings = all.split(" ");
        type=strings[0];
        value=strings[1];
    }
    public int getComV() {
        return sortMap.get(value);
    }


    @Override
    public int compareTo(Object o) {
        Card c=(Card) o;
        if(getComV()!=c.getComV()) {
            return Integer.compare(getComV(),c.getComV());
        }else if(getComV()!=10){
            return Integer.compare(Card.sortMap.get(type),Card.sortMap.get(c.type));
        }else{
            return Integer.compare(tempV.get(value),tempV.get(c.value));
        }
    }

    @Override
    public String toString() {
        return type+value;
    }
}

class Handle{
    List<Card> cards;
    int value;

    public Handle() {
        this.cards =new ArrayList<>();
    }

    public void add(Card c){
        cards.add(c);
    }

    public int getValue() {
        int res=0;
        cards.sort(Card::compareTo);
        for (int i = cards.size() - 1; i >= 0; i--) {
            Card card = cards.get(i);
            if(card.getComV()!=1) res+=card.getComV();
            else{
                if(res+11<=21&&res+11+i<=21) res+=11;
                else res+=1;
            }
        }
        value=res;
        return value;
    }

    public int size(){
        return cards.size();
    }

    @Override
    public String toString() {
        StringBuilder sb=new StringBuilder();
        cards.sort(Card::compareTo);
        for (Card card : cards) {
            sb.append(card.toString()).append(" ");
        }
        return sb.toString();
    }
}

abstract class Employee implements Comparable{
    String firstName;
    String lastName;
    String socialSecurityNumber;

    abstract double earning();

    String StrEarning(){
        return String.format("%.2f", earning());
    }

    public Employee(String[] args) {
        assert args.length>=3;
        firstName=args[0];
        lastName=args[1];
        socialSecurityNumber=args[2];
    }

    @Override
    public String toString() {
        return "firstName:"+firstName+"; "
                +"lastName:"+lastName+"; "
                +"socialSecurityNumber:"+socialSecurityNumber+"; "
                +"earning:"+this.StrEarning();
    }

    @Override
    public int compareTo(Object o) {
        return Double.compare(earning(),((Employee)o).earning());
    }
}

class SalaridEmployee extends Employee{
    double weeklySalary;

    public SalaridEmployee(String[] args) {
        super(args);
        weeklySalary= Double.parseDouble(args[3]);
    }

    @Override
    double earning() {
        return weeklySalary*4;
    }
}

class HourlyEmployee extends Employee{
    double wage;
    double hour;

    public HourlyEmployee(String[] args) {
        super(args);
        wage= Double.parseDouble(args[3]);
        hour= Double.parseDouble(args[4]);
    }

    @Override
    double earning() {
        return wage*hour;
    }
}

class CommisionEmployee extends Employee{
    double grossSales;
    double commissionRate;

    public CommisionEmployee(String[] args) {
        super(args);
        grossSales= Double.parseDouble(args[3]);
        commissionRate= Double.parseDouble(args[4]);
    }

    @Override
    double earning() {
        return grossSales*commissionRate;
    }
}

class basePlusCommisionEmployee extends CommisionEmployee{
    double baseSalary;

    public basePlusCommisionEmployee(String[] args) {
        super(args);
        baseSalary= Double.parseDouble(args[5]);
    }

    @Override
    double earning() {
        return grossSales*commissionRate+baseSalary;
    }
}

class Person implements Serializable {

    private static final long serialVersionUID = 1L;
    private String name;
    private String gender;
    private LocalDate birthday;
    private String biography;

    public Person() {

    }

    public Person(String name, String gender, String biography,
                  int year, int month, int day) {
        this.name = name;
        this.gender = gender;
        this.biography = biography;
        this.birthday = LocalDate.of(year, month, day);
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public LocalDate getBirthday() {
        return birthday;
    }

    public void setBirthday(LocalDate birthday) {
        this.birthday = birthday;
    }

    public String getBiography() {
        return biography;
    }

    public void setBiography(String biography) {
        this.biography = biography;
    }

    @Override
    public String toString() {
        return "name: " + name + " , gender: " + gender + " , birthday: "
                + birthday + " , biography: " + biography;
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
        int[] dp = new int[volume + 8], hash = new int[9003];
        for (int k = 0; k < nums; k++) {
            for (int i = volume; i > 0; i--) {
//				hash[max(0, dp[i])] = 1;
//				if (i >= weight[k]) hash[max(0, dp[i - weight[k]] + value[k])] = 1;
                if (i >= weight[k]) dp[i] = max(dp[i - weight[k]] + value[k], dp[i]);
            }
        }
        return dp[volume];
    }

    public long ZeroOnePackCount(int v) {
        //得到总价值为value时选择的方案数，容量不做限制
        //dp【j】【i】为考虑前j个物品，价值为i时的方案数，j维度通过滚动for删除
        //dp[i]=i<value[j]?  dp[i] : dp[i]+dp[i-value[j]]
        //与上面相同，应倒序，保证dpi是从没有选择过物品j的状态转移而来
        long[] dp = new long[v];
        dp[0] = 1;
        for (int j = 0; j < nums; j++) {
            for (int i = v; i >= 0; i--) {
                dp[i] = i < value[j] ? dp[i] : dp[i] + dp[i - value[j]];
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
            int k = multiple[i] & -multiple[i];
            while (true) {
                if (k <= multiple[i]) {
                    w.add(i, k * weight[i]);
                    v.add(i, k * value[i]);
                    multiple[i] -= k;
                    k *= 2;
                } else {
                    k = multiple[i];
                    if (k != 0) {
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

class Book implements Serializable {

    private static final long serialVersionUID = 1L;

    private String name;
    private Person author;
    private int price;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Person getAuthor() {
        return author;
    }

    public void setAuthor(Person author) {
        this.author = author;
    }

    public int getPrice() {
        return price;
    }

    public void setPrice(int price) {
        this.price = price;
    }

    public Book() {

    }

    public Book(String name, Person author, int price) {
        this.name = name;
        this.author = author;
        this.price = price;
    }

    @Override
    public String toString() {
        return "name: " + name + "\nauthor: " + author + "\nprice: " + price;
    }

}

class myComplex {
    private final double re; // the real part
    private final double im; // the imaginary part

    // create a new object with the given real and imaginary parts
    public myComplex(double real, double imag) {
        re = real;
        im = imag;
    }
    //返回单位圆分成n份后，k刻度（第k个）点的虚数系下的坐标，fft用
    public static myComplex omega(int n, int k){
        return new myComplex(Math.cos(2*Math.PI*k/n),Math.sin(2*Math.PI*k/n));
    }
    // return a string representation of the invoking Complex object
    @Override
    public String toString() {
        return String.format("%.1f", re)+
                (im>=0?"+":"")+
                String.format("%.1fi", im);
    }

    // return abs/modulus/magnitude
    public double abs() {
        return Math.hypot(re, im);
    }

    // return angle/phase/argument, normalized target be between -pi and pi
    public double phase() {
        return Math.atan2(im, re);
    }

    // return a new Complex object whose value is (this + b)
    public myComplex plus(myComplex b) {
        myComplex a = this; // invoking object
        double real = a.re + b.re;
        double imag = a.im + b.im;
        return new myComplex(real, imag);
    }

    // return a new Complex object whose value is (this - b)
    public myComplex minus(myComplex b) {
        myComplex a = this;
        double real = a.re - b.re;
        double imag = a.im - b.im;
        return new myComplex(real, imag);
    }

    // return a new Complex object whose value is (this * b)
    public myComplex multiple(myComplex b) {
        myComplex a = this;
        double real = a.re * b.re - a.im * b.im;
        double imag = a.re * b.im + a.im * b.re;
        return new myComplex(real, imag);
    }

    // scalar multiplication
    // return a new object whose value is (this * alpha)
    public myComplex multiple(double alpha) {
        return new myComplex(alpha * re, alpha * im);
    }

    // return a new object whose value is (this * alpha)
    public myComplex scale(double alpha) {
        return new myComplex(alpha * re, alpha * im);
    }

    // return a new Complex object whose value is the conjugate of this
    public myComplex conjugate() {
        return new myComplex(re, -im);
    }

    // return a new Complex object whose value is the reciprocal of this
    public myComplex reciprocal() {
        double scale = re * re + im * im;
        return new myComplex(re / scale, -im / scale);
    }

    // return the real or imaginary part
    public double re() {
        return re;
    }

    public double im() {
        return im;
    }

    // return a / b
    public myComplex divides(myComplex b) {
        myComplex a = this;
        return a.multiple(b.reciprocal());
    }

    // return a new Complex object whose value is the complex exponential of
    // this
    public myComplex exp() {
        return new myComplex(Math.exp(re) * Math.cos(im), Math.exp(re) * Math.sin(im));
    }

    // return a new Complex object whose value is the complex sine of this
    public myComplex sin() {
        return new myComplex(Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im));
    }

    // return a new Complex object whose value is the complex cosine of this
    public myComplex cos() {
        return new myComplex(Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im));
    }

    // return a new Complex object whose value is the complex tangent of this
    public myComplex tan() {
        return sin().divides(cos());
    }

    // a static version of plus
    public static myComplex plus(myComplex a, myComplex b) {
        double real = a.re + b.re;
        double imag = a.im + b.im;
        myComplex sum = new myComplex(real, imag);
        return sum;
    }

    // See Section 3.3.
    @Override
    public boolean equals(Object x) {
        if (x == null)
            return false;
        if (this.getClass() != x.getClass())
            return false;
        myComplex that = (myComplex) x;
        return (this.re == that.re) && (this.im == that.im);
    }

    // See Section 3.3.
    @Override
    public int hashCode() {
        return Objects.hash(re, im);
    }
}
