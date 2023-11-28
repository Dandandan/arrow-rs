(function() {var type_impls = {
"arrow_array":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-GenericByteBuilder%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#35-162\">source</a><a href=\"#impl-GenericByteBuilder%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"arrow_array/types/trait.ByteArrayType.html\" title=\"trait arrow_array::types::ByteArrayType\">ByteArrayType</a>&gt; <a class=\"struct\" href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\">GenericByteBuilder</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.new\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#37-39\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.new\" class=\"fn\">new</a>() -&gt; Self</h4></section></summary><div class=\"docblock\"><p>Creates a new <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\"><code>GenericByteBuilder</code></a>.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.with_capacity\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#47-55\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.with_capacity\" class=\"fn\">with_capacity</a>(item_capacity: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>, data_capacity: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>) -&gt; Self</h4></section></summary><div class=\"docblock\"><p>Creates a new <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\"><code>GenericByteBuilder</code></a>.</p>\n<ul>\n<li><code>item_capacity</code> is the number of items to pre-allocate.\nThe size of the preallocated buffer of offsets is the number of items plus one.</li>\n<li><code>data_capacity</code> is the total number of bytes of data to pre-allocate\n(for all items, not per item).</li>\n</ul>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.new_from_buffer\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#62-79\">source</a><h4 class=\"code-header\">pub unsafe fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.new_from_buffer\" class=\"fn\">new_from_buffer</a>(\n    offsets_buffer: <a class=\"struct\" href=\"arrow_buffer/buffer/mutable/struct.MutableBuffer.html\" title=\"struct arrow_buffer::buffer::mutable::MutableBuffer\">MutableBuffer</a>,\n    value_buffer: <a class=\"struct\" href=\"arrow_buffer/buffer/mutable/struct.MutableBuffer.html\" title=\"struct arrow_buffer::buffer::mutable::MutableBuffer\">MutableBuffer</a>,\n    null_buffer: <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;<a class=\"struct\" href=\"arrow_buffer/buffer/mutable/struct.MutableBuffer.html\" title=\"struct arrow_buffer::buffer::mutable::MutableBuffer\">MutableBuffer</a>&gt;\n) -&gt; Self</h4></section></summary><div class=\"docblock\"><p>Creates a new  <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\"><code>GenericByteBuilder</code></a> from buffers.</p>\n<h5 id=\"safety\"><a href=\"#safety\">Safety</a></h5>\n<p>This doesn’t verify buffer contents as it assumes the buffers are from existing and\nvalid <a href=\"arrow_array/array/byte_array/struct.GenericByteArray.html\" title=\"struct arrow_array::array::byte_array::GenericByteArray\"><code>GenericByteArray</code></a>.</p>\n</div></details><section id=\"method.next_offset\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#82-84\">source</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.next_offset\" class=\"fn\">next_offset</a>(&amp;self) -&gt; T::<a class=\"associatedtype\" href=\"arrow_array/types/trait.ByteArrayType.html#associatedtype.Offset\" title=\"type arrow_array::types::ByteArrayType::Offset\">Offset</a></h4></section><details class=\"toggle method-toggle\" open><summary><section id=\"method.append_value\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#92-96\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.append_value\" class=\"fn\">append_value</a>(&amp;mut self, value: impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.AsRef.html\" title=\"trait core::convert::AsRef\">AsRef</a>&lt;T::<a class=\"associatedtype\" href=\"arrow_array/types/trait.ByteArrayType.html#associatedtype.Native\" title=\"type arrow_array::types::ByteArrayType::Native\">Native</a>&gt;)</h4></section></summary><div class=\"docblock\"><p>Appends a value into the builder.</p>\n<h5 id=\"panics\"><a href=\"#panics\">Panics</a></h5>\n<p>Panics if the resulting length of <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#method.values_slice\" title=\"method arrow_array::builder::generic_bytes_builder::GenericByteBuilder::values_slice\"><code>Self::values_slice</code></a> would exceed <code>T::Offset::MAX</code></p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.append_option\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#100-105\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.append_option\" class=\"fn\">append_option</a>(&amp;mut self, value: <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.AsRef.html\" title=\"trait core::convert::AsRef\">AsRef</a>&lt;T::<a class=\"associatedtype\" href=\"arrow_array/types/trait.ByteArrayType.html#associatedtype.Native\" title=\"type arrow_array::types::ByteArrayType::Native\">Native</a>&gt;&gt;)</h4></section></summary><div class=\"docblock\"><p>Append an <code>Option</code> value into the builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.append_null\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#109-112\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.append_null\" class=\"fn\">append_null</a>(&amp;mut self)</h4></section></summary><div class=\"docblock\"><p>Append a null value into the builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.finish\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#115-126\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.finish\" class=\"fn\">finish</a>(&amp;mut self) -&gt; <a class=\"struct\" href=\"arrow_array/array/byte_array/struct.GenericByteArray.html\" title=\"struct arrow_array::array::byte_array::GenericByteArray\">GenericByteArray</a>&lt;T&gt;</h4></section></summary><div class=\"docblock\"><p>Builds the <a href=\"arrow_array/array/byte_array/struct.GenericByteArray.html\" title=\"struct arrow_array::array::byte_array::GenericByteArray\"><code>GenericByteArray</code></a> and reset this builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.finish_cloned\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#129-141\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.finish_cloned\" class=\"fn\">finish_cloned</a>(&amp;self) -&gt; <a class=\"struct\" href=\"arrow_array/array/byte_array/struct.GenericByteArray.html\" title=\"struct arrow_array::array::byte_array::GenericByteArray\">GenericByteArray</a>&lt;T&gt;</h4></section></summary><div class=\"docblock\"><p>Builds the <a href=\"arrow_array/array/byte_array/struct.GenericByteArray.html\" title=\"struct arrow_array::array::byte_array::GenericByteArray\"><code>GenericByteArray</code></a> without resetting the builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.values_slice\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#144-146\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.values_slice\" class=\"fn\">values_slice</a>(&amp;self) -&gt; &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u8.html\">u8</a>] <a href=\"#\" class=\"tooltip\" data-notable-ty=\"&amp;[u8]\">ⓘ</a></h4></section></summary><div class=\"docblock\"><p>Returns the current values buffer as a slice</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.offsets_slice\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#149-151\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.offsets_slice\" class=\"fn\">offsets_slice</a>(&amp;self) -&gt; &amp;[T::<a class=\"associatedtype\" href=\"arrow_array/types/trait.ByteArrayType.html#associatedtype.Offset\" title=\"type arrow_array::types::ByteArrayType::Offset\">Offset</a>]</h4></section></summary><div class=\"docblock\"><p>Returns the current offsets buffer as a slice</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.validity_slice\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#154-156\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.validity_slice\" class=\"fn\">validity_slice</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u8.html\">u8</a>]&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the current null buffer as a slice</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.validity_slice_mut\" class=\"method\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#159-161\">source</a><h4 class=\"code-header\">pub fn <a href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html#tymethod.validity_slice_mut\" class=\"fn\">validity_slice_mut</a>(&amp;mut self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;mut [<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u8.html\">u8</a>]&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the current null buffer as a mutable slice</p>\n</div></details></div></details>",0,"arrow_array::builder::generic_bytes_builder::GenericStringBuilder","arrow_array::builder::generic_bytes_builder::GenericBinaryBuilder"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-GenericByteBuilder%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#164-173\">source</a><a href=\"#impl-Debug-for-GenericByteBuilder%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"arrow_array/types/trait.ByteArrayType.html\" title=\"trait arrow_array::types::ByteArrayType\">ByteArrayType</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\">GenericByteBuilder</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#165-172\">source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/fmt/type.Result.html\" title=\"type core::fmt::Result\">Result</a></h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","arrow_array::builder::generic_bytes_builder::GenericStringBuilder","arrow_array::builder::generic_bytes_builder::GenericBinaryBuilder"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Default-for-GenericByteBuilder%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#175-179\">source</a><a href=\"#impl-Default-for-GenericByteBuilder%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"arrow_array/types/trait.ByteArrayType.html\" title=\"trait arrow_array::types::ByteArrayType\">ByteArrayType</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/default/trait.Default.html\" title=\"trait core::default::Default\">Default</a> for <a class=\"struct\" href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\">GenericByteBuilder</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.default\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#176-178\">source</a><a href=\"#method.default\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/default/trait.Default.html#tymethod.default\" class=\"fn\">default</a>() -&gt; Self</h4></section></summary><div class='docblock'>Returns the “default value” for a type. <a href=\"https://doc.rust-lang.org/nightly/core/default/trait.Default.html#tymethod.default\">Read more</a></div></details></div></details>","Default","arrow_array::builder::generic_bytes_builder::GenericStringBuilder","arrow_array::builder::generic_bytes_builder::GenericBinaryBuilder"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Extend%3COption%3CV%3E%3E-for-GenericByteBuilder%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#213-220\">source</a><a href=\"#impl-Extend%3COption%3CV%3E%3E-for-GenericByteBuilder%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"arrow_array/types/trait.ByteArrayType.html\" title=\"trait arrow_array::types::ByteArrayType\">ByteArrayType</a>, V: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.AsRef.html\" title=\"trait core::convert::AsRef\">AsRef</a>&lt;T::<a class=\"associatedtype\" href=\"arrow_array/types/trait.ByteArrayType.html#associatedtype.Native\" title=\"type arrow_array::types::ByteArrayType::Native\">Native</a>&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html\" title=\"trait core::iter::traits::collect::Extend\">Extend</a>&lt;<a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;V&gt;&gt; for <a class=\"struct\" href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\">GenericByteBuilder</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.extend\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#215-219\">source</a><a href=\"#method.extend\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html#tymethod.extend\" class=\"fn\">extend</a>&lt;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a>&lt;Item = <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;V&gt;&gt;&gt;(&amp;mut self, iter: I)</h4></section></summary><div class='docblock'>Extends a collection with the contents of an iterator. <a href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html#tymethod.extend\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.extend_one\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"https://doc.rust-lang.org/nightly/src/core/iter/traits/collect.rs.html#376\">source</a><a href=\"#method.extend_one\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html#method.extend_one\" class=\"fn\">extend_one</a>(&amp;mut self, item: A)</h4></section></summary><span class=\"item-info\"><div class=\"stab unstable\"><span class=\"emoji\">🔬</span><span>This is a nightly-only experimental API. (<code>extend_one</code>)</span></div></span><div class='docblock'>Extends a collection with exactly one element.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.extend_reserve\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"https://doc.rust-lang.org/nightly/src/core/iter/traits/collect.rs.html#384\">source</a><a href=\"#method.extend_reserve\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html#method.extend_reserve\" class=\"fn\">extend_reserve</a>(&amp;mut self, additional: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>)</h4></section></summary><span class=\"item-info\"><div class=\"stab unstable\"><span class=\"emoji\">🔬</span><span>This is a nightly-only experimental API. (<code>extend_one</code>)</span></div></span><div class='docblock'>Reserves capacity in a collection for the given number of additional elements. <a href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.Extend.html#method.extend_reserve\">Read more</a></div></details></div></details>","Extend<Option<V>>","arrow_array::builder::generic_bytes_builder::GenericStringBuilder","arrow_array::builder::generic_bytes_builder::GenericBinaryBuilder"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ArrayBuilder-for-GenericByteBuilder%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#181-211\">source</a><a href=\"#impl-ArrayBuilder-for-GenericByteBuilder%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"arrow_array/types/trait.ByteArrayType.html\" title=\"trait arrow_array::types::ByteArrayType\">ByteArrayType</a>&gt; <a class=\"trait\" href=\"arrow_array/builder/trait.ArrayBuilder.html\" title=\"trait arrow_array::builder::ArrayBuilder\">ArrayBuilder</a> for <a class=\"struct\" href=\"arrow_array/builder/generic_bytes_builder/struct.GenericByteBuilder.html\" title=\"struct arrow_array::builder::generic_bytes_builder::GenericByteBuilder\">GenericByteBuilder</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.len\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#183-185\">source</a><a href=\"#method.len\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.len\" class=\"fn\">len</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the number of binary slots in the builder</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.finish\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#188-190\">source</a><a href=\"#method.finish\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.finish\" class=\"fn\">finish</a>(&amp;mut self) -&gt; <a class=\"type\" href=\"arrow_array/array/type.ArrayRef.html\" title=\"type arrow_array::array::ArrayRef\">ArrayRef</a></h4></section></summary><div class=\"docblock\"><p>Builds the array and reset this builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.finish_cloned\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#193-195\">source</a><a href=\"#method.finish_cloned\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.finish_cloned\" class=\"fn\">finish_cloned</a>(&amp;self) -&gt; <a class=\"type\" href=\"arrow_array/array/type.ArrayRef.html\" title=\"type arrow_array::array::ArrayRef\">ArrayRef</a></h4></section></summary><div class=\"docblock\"><p>Builds the array without resetting the builder.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_any\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#198-200\">source</a><a href=\"#method.as_any\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.as_any\" class=\"fn\">as_any</a>(&amp;self) -&gt; &amp;dyn <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/any/trait.Any.html\" title=\"trait core::any::Any\">Any</a></h4></section></summary><div class=\"docblock\"><p>Returns the builder as a non-mutable <code>Any</code> reference.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_any_mut\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#203-205\">source</a><a href=\"#method.as_any_mut\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.as_any_mut\" class=\"fn\">as_any_mut</a>(&amp;mut self) -&gt; &amp;mut dyn <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/any/trait.Any.html\" title=\"trait core::any::Any\">Any</a></h4></section></summary><div class=\"docblock\"><p>Returns the builder as a mutable <code>Any</code> reference.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.into_box_any\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/generic_bytes_builder.rs.html#208-210\">source</a><a href=\"#method.into_box_any\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#tymethod.into_box_any\" class=\"fn\">into_box_any</a>(self: <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/alloc/boxed/struct.Box.html\" title=\"struct alloc::boxed::Box\">Box</a>&lt;Self&gt;) -&gt; <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/alloc/boxed/struct.Box.html\" title=\"struct alloc::boxed::Box\">Box</a>&lt;dyn <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/any/trait.Any.html\" title=\"trait core::any::Any\">Any</a>&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the boxed builder as a box of <code>Any</code>.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.is_empty\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arrow_array/builder/mod.rs.html#240-242\">source</a><a href=\"#method.is_empty\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"arrow_array/builder/trait.ArrayBuilder.html#method.is_empty\" class=\"fn\">is_empty</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Returns whether number of array slots is zero</div></details></div></details>","ArrayBuilder","arrow_array::builder::generic_bytes_builder::GenericStringBuilder","arrow_array::builder::generic_bytes_builder::GenericBinaryBuilder"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()