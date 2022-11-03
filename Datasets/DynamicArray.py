
##
## This class describes an dynamically growing array. It is especially useful
## when you want to insert an item at index n but you don't know upfront what
## the largest index n will be. DynamicArray allocates the necessary space so
## that you can insert at any index.
## Inbetween the inserted values, the array contains None items.
##
class DynamicArray:
	def __init__(self):
		self.array = []
		self.maxIndex = -1
		self.capacity = 0

	def insert(self, index, item):
		if index < 0: index = self.maxIndex + 1 + index
		self.array.insert(index, item)
		self.capacity += 1
		self.maxIndex += 1

	def resizeArray(self):
		if self.capacity == 0: self.capacity = 1
		else: self.capacity *= 2
		newArr = [None] * self.capacity
		for i in range(0, len(self.array)):
			newArr[i] = self.array[i]
		self.array = newArr

	def toList(self):
		return self.array[:self.maxIndex+1]

	##
	## Returns length up to item inserted at largest index
	##
	## :returns:   { length up to item inserted at largest index }
	## :rtype:     { int }
	##
	def size(self):
		return self.maxIndex+1

	def __iter__(self):
		return self.toList().__iter__()

	def __getitem__(self, subscript):
		if (subscript < 0 or subscript >= self.capacity):
			raise ValueError("Index out of bounds!")
		return self.array[subscript]

	def __len__(self):
		return self.size()

	def __setitem__(self, index, value):
		while (index >= self.capacity): # nice
			self.resizeArray()
		self.array[index] = value
		self.maxIndex = max(self.maxIndex, index)

	@property # for JSON serialization
	def __dict__(self):
		return self.toList()
