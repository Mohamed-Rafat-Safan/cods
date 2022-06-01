// تحت اشراف  
 //   Mohamed Rafat Safan

package heab_sort;





public class Heab_Sort {
 
    public void Stor(int[] arr)
    {
        int n = arr.length ;
        // Build max heap
      for (int i = n / 2 - 1; i >= 0; i--)
      {
        Heapify(arr , n ,  i);
      }
      
      //Heap sort 
      for(int i= n-1 ; i>=0 ; i--)
      {
         int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        
         // Heapify root element
        Heapify(arr, i, 0);
      }
    }
    
    public void Heapify(int[] arr ,int n , int i)
    {
        int max = i ;
        int l = (2*i)+1 ;
        int r = (2*i)+2 ;
        if(l<n && arr[l]>=arr[max])
            max = l ;
        
        if(r<n && arr[r]>=arr[max])
            max = r ;
        
        if(max !=i)
        {
            int temp = arr[i] ;
            arr[i] = arr[max] ;
            arr[max] = temp ;
            
            
             Heapify(arr , n , max);
        }
            
        
        
    }
    
   
    public void display(int[] arr)
    {
        for(int i=0 ; i<arr.length ; i++)
        {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    
    
    
    public static void main(String[] args) {
        // TODO code application logic here
        int arr[] = { 1, 12, 9, 5, 6, 10 };
        Heab_Sort ob = new Heab_Sort() ;
        
        
        ob.display(arr);
        
        ob.Stor(arr);
        ob.display(arr);

    }
    
}
