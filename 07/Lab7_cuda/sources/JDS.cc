// acmicpc number: JDS
#include <bits/stdc++.h>
using namespace std;

#define FOR(i, n) for (int i = 0; i < (n); i++)
#define REP(i, a, b) for (int i=(a); i < (b); i++)

typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<ll> vl;
typedef vector<vl> vll;
typedef vector<int> vi;
typedef vector<vi> vii;

ll a, b;

struct JDS_T
{
  float *data;
  int data_size;
  int *jds_col_idx;
  int jds_col_idx_size;
  int *jds_row_idx;
  int jds_row_idx_size;
  int *jds_row_ptr;
  int jds_row_ptr_size;
  int *jds_t_col_ptr;
  int jds_t_col_ptr_size;
};

int main()
{
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  cin >> a >> b;

  float *hostA = (float *)malloc(a * b * sizeof(float));
  for (int i = 0; i < a; i++)
  {
    for (int j = 0; j < b; j++)
    {
      cin >> hostA[i * b + j];
    }
  }

  int numARows = a;
  int numAColumns = b;

  vector<std::pair<int, int>> mat_count;
  int total_non_zero = 0;
  for (int i = 0; i < numARows; i++)
  {
    int count = 0;
    for (int j = 0; j < numAColumns; j++)
    {
      if (hostA[i * numAColumns + j] != 0)
      {
        count++;
        total_non_zero++;
      }
    }
    mat_count.push_back(std::make_pair(count, i));
  }

  std::sort(mat_count.begin(), mat_count.end(), std::greater<std::pair<int, int>>());

  vector<float> data;
  vector<int> jds_col_idx;
  vector<int> jds_row_idx;
  vector<int> jds_row_ptr;

  for (auto& i : mat_count)
  {
    jds_row_ptr.push_back(data.size());
    jds_row_idx.push_back(i.second);
    for (int j = 0; j < numAColumns; j++)
    {
      if (hostA[i.second * numAColumns + j] != 0)
      {
        data.push_back(hostA[i.second * numAColumns + j]);
        jds_col_idx.push_back(j);
      }
    }
  }
  jds_row_ptr.push_back(data.size());

  vector<float> data2;
  vector<int> jds_col_idx2;
  vector<int> jds_col_ptr;
  // transpose??
  int tc = 0;
  // this is the maximum block size required
  int max_count = 0;
  // this is the number of rows (transposed)
  for (int i = 0; i < mat_count[0].first; i++)
  {
    // find how many pairs have the first value bigger than i
    jds_col_ptr.push_back(tc);

    // count is the number of columns for this row i
    int count = 0;
    for (int j = 0; j < mat_count.size(); j++)
    {
      if (mat_count[j].first > i)
      {
        count++;
      }
      else
      {
        // it is sorted, so we can break
        break;
      }
    }
    if (count > max_count)
    {
      max_count = count;
    }
    tc += count;

    // retrieve the elements from data
    for (int k = 0; k < count; k++)
    {
      data2.push_back(data[jds_row_ptr[k] + i]);
      jds_col_idx2.push_back(jds_col_idx[jds_row_ptr[k] + i]);
    }
  }
  jds_col_ptr.push_back(tc);

  JDS_T jds;
  jds.data = data2.data();
  jds.data_size = data2.size();
  jds.jds_col_idx = jds_col_idx2.data();
  jds.jds_col_idx_size = jds_col_idx2.size();
  jds.jds_row_idx = jds_row_idx.data();
  jds.jds_row_idx_size = jds_row_idx.size();
  jds.jds_row_ptr = jds_row_ptr.data();
  jds.jds_row_ptr_size = jds_row_ptr.size();
  jds.jds_t_col_ptr = jds_col_ptr.data();
  jds.jds_t_col_ptr_size = jds_col_ptr.size();

  cout << "max_count: " << max_count << endl;

  cout << "data: " << jds.data_size << endl;
  for (int i = 0; i < jds.data_size; i++)
  {
    cout << jds.data[i] << " ";
  }
  cout << endl << endl;

  cout << "jds_col_idx: " << jds.jds_col_idx_size << endl;
  for (int i = 0; i < jds.jds_col_idx_size; i++)
  {
    cout << jds.jds_col_idx[i] << " ";
  }
  cout << endl << endl;

  cout << "jds_row_idx: " << jds.jds_row_idx_size << endl;
  for (int i = 0; i < jds.jds_row_idx_size; i++)
  {
    cout << jds.jds_row_idx[i] << " ";
  }
  cout << endl << endl;

  cout << "jds_row_ptr: " << jds.jds_row_ptr_size << endl;
  for (int i = 0; i < jds.jds_row_ptr_size; i++)
  {
    cout << jds.jds_row_ptr[i] << " ";
  }
  cout << endl << endl;

  cout << "jds_t_col_ptr: " << jds.jds_t_col_ptr_size << endl;
  for (int i = 0; i < jds.jds_t_col_ptr_size; i++)
  {
    cout << jds.jds_t_col_ptr[i] << " ";
  }
  cout << endl;
}
