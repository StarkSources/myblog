---
layout: post
title: "Min-Max-Sorting"
author: sal
categories: [ Useful, tutorial ]
tags: coding
image: assets/images/16.jpg
---

{% highlight ruby linenos %}
public class MinMax {
    public static void main(String args[]){
        int n;
        Scanner s = new Scanner(System.in);    
        n= s.nextInt();
        Integer array[] = new Integer[n];
        int sorted[]= new int[n];
        int count=0;
        for(int i=0;i<n;i++){
            array[i]=s.nextInt();
            if(array[i]==0){
                count=count+1;
                
            }
        }
       int p=0,q=n-1,k;
        int max,min;
        int M=0,m=0;    
        
        for(int j=0;j<=n/2;j++){
            min=max=array[0];
            M=0;m=0;
            int z=0;
           while(min==0 &&(z<n)){
            min=array[z];
            z++;
            }
           z=0;
           while(max==0 &&(z<n)){
            max=array[z];
            z++;
            }
            for( k=0;k<n;k++){
                if(array[k]>=max && array[k]!=0){
                    max=array[k];
                    M=k;
                }
                  if((array[k]<=min && array[k]!=0)||(array[k]<min && count!=0)){  
                    min=array[k];
                    m=k;
                    if(array[k]==0){
                         min=array[k];
                         m=k;
                        count=count-1;
                    }
                 
                }
                
            }
                sorted[q]=max;
                q=q-1;
               sorted[p]=min;
                p=p+1;  
                array[M]=0;
                array[m]=0;
              int count2=0;
        for(int i=0;i<n;i++){
            if(array[i]==0){
                count2=count2+1;
            }           
      }
        if(count2==n){
            break;
        }
    }
        System.out.println("after sorting the numbers are: ");
        for(int i=0;i<n;i++){
            System.out.print(sorted[i]+" ");
        }

    }
}
{% endhighlight %}
