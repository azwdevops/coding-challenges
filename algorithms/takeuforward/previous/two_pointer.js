const maximumCardPoints = function (nums, k) {
  let leftSum = 0;
  let rightSum = 0;
  let maxSum = 0;

  for (let i = 0; i < k; i++) {
    leftSum += nums[i];
    maxSum = leftSum;
  }

  let rightIndex = nums.length - 1;

  for (let i = k - 1; i >= 0; i--) {
    leftSum -= nums[i];
    rightSum += nums[rightIndex];
    rightIndex -= 1;

    maxSum = Math.max(maxSum, leftSum + rightSum);
  }

  return maxSum;
};

// minimum window substring where given s and t all chars in t and their frequency appear in s
const minWindowSubstring = function (s, t) {
  if (s === "" || t === "") {
    return "";
  }
  const freq_t = {};
  for (const char of t) {
    if (freq_t[char]) {
      freq_t[char] += 1;
    } else {
      freq_t[char] = 1;
    }
  }
  const required = freq_t.length; // required chars to make substring of s equal to t
  let left = 0;
  let right = 0;

  const window_s = {};

  let formed = 0;

  let min_window = "";

  while (right < s.length) {
    const char = s[right];

    if (window_s[char]) {
      window_s[char] += 1;
    } else {
      window_s[char] = 1;
    }
    if (char in freq_t && window_s[char] === freq_t[char]) {
      formed += 1;
    }

    while (left <= right && formed === required) {
      if (right - left + 1 < min_window.length) {
        min_window = s.slice(left, right + 1);
      }
      window_s[s[left]] -= 1;
      const currentChar = s[left];
      if (currentChar in freq_t && window_s[currentChar] < freq_t[currentChar]) {
        formed -= 1;
      }
      left += 1;
    }
    right += 1;
  }

  return min_window;
};

// minimum window subsequence
const minWindowSubsequence = function (S, T) {
  const m = S.length;
  const n = T.length;
  let min_length = Infinity;
  let start_index = -1;

  let i = 0;
  while (i < m) {
    // find the first character of T in s
    if (S[i] === T[0]) {
      let j = 0;
      let k = i;

      // try to find the entire subsequence T in S starting from S[i]
      while (k < m && j < n) {
        if (S[k] == T[j]) {
          j += 1;
        }
        k += 1;
      }

      // if we found the entire subsequence
      if (j == n) {
        // now we try to shrink the window by moving the left pointer
        const end = k;
        j -= 1;
        k -= 1;

        while (j >= 0) {
          if (S[k] === T[j]) {
            j -= 1;
          }
          k -= 1;
        }
        if (end - k - 1 < min_length) {
          min_length = end - k - 1;
          start_index = k + 1;
        }
      }
    }
    i += 1;
  }

  return start_index === -1 ? "" : S.slice(start_index, start_index + min_length);
};
