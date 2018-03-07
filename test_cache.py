import unittest

from PyCircuit import Clock, TestVector
from cache import Cache, Set


class TestCache(unittest.TestCase):
    def test_Cache(self):
        clk = Clock()
        rst = TestVector(0, 1)
        i_p_addr = TestVector(0, 25)
        i_p_byte_en = TestVector(0, 4)
        i_p_writedata = TestVector(0, 32)
        i_p_read = TestVector(0, 1)
        i_p_write = TestVector(0, 1)
        i_m_readdata = TestVector(0, 128)
        i_m_readdata_valid = TestVector(0, 1)
        i_m_waitrequest = TestVector(0, 1)
        # input wire [24:0]  i_p_addr;
        # input wire [3:0]   i_p_byte_en;
        # input wire [31:0]  i_p_writedata;
        # input wire         i_p_read, i_p_write;
        # output reg [31:0]  o_p_readdata;
        # output reg         o_p_readdata_valid;
        # output wire        o_p_waitrequest;
        #
        # output reg [25:0]  o_m_addr;
        # output wire [3:0]  o_m_byte_en;
        # output reg [127:0] o_m_writedata;
        # output reg         o_m_read, o_m_write;
        # input wire [127:0] i_m_readdata;
        # input wire         i_m_readdata_valid;
        # input wire         i_m_waitrequest;
        o_p_readdata, o_p_readdata_valid, o_p_waitrequest, o_m_addr, o_m_byte_en, o_m_writedata, o_m_read, o_m_write = Cache(
            clk,
            rst,
            i_p_addr,
            i_p_byte_en,
            i_p_writedata,
            i_p_read,
            i_p_write,

            i_m_readdata,
            i_m_readdata_valid,
            i_m_waitrequest)
        # self.assertVectorEqual(q, d)
        # self.assertCurrent(q)
        # print(calcAreaDelay(a.concat(d).concat(mem_wr)))
        # for i in range(2 ** SZ):
        #     a[:] = i
        #     self.assertVectorEqual(q, 0)
        #     self.assertCurrent(q)
        # for i in range(2 ** SZ):
        #     d[:] = i
        #     a[:] = i
        #     mem_wr[:] = 1
        #     self.assertVectorEqual(q, d)
        #     self.assertCurrent(q)
        #     mem_wr[:] = 0
        #     self.assertVectorEqual(q, d)
        #     self.assertCurrent(q)
        # for i in range(2 ** SZ):
        #     a[:] = i
        #     self.assertVectorEqual(q, i)
        #     self.assertCurrent(q)

    def test_Set(self):
        clk = Clock()
        cache_entry = 14
        entry = TestVector(0, cache_entry)
        o_tag = TestVector(0, 23 - cache_entry)
        writedata = TestVector(0, 128)
        byte_en = TestVector(0, 4)
        write = TestVector(0, 1)
        word_en = TestVector(0, 4)
        read_miss = TestVector(0, 1)
        readdata, wb_addr, hit, modify, miss, valid = Set(clk,
                                                          entry,
                                                          o_tag,
                                                          writedata,
                                                          byte_en,
                                                          write,
                                                          word_en,
                                                          read_miss)
